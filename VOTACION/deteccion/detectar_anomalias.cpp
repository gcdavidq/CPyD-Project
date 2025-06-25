/* detectar_anomalias_corregido.cpp */

#include "VOTACION/deteccion/detectar_anomalias.hpp"
#include "VOTACION/common/estructura_votos.hpp"
#include <omp.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <functional>
#include <iostream>

using namespace std;

namespace deteccion {

    // Implementación corregida del detector de anomalías en CPU usando OpenMP
    ResultadoDeteccion detectarAnomaliasCPU(const std::vector<Voto>& votos, int num_hilos) {
        int max_hilos = omp_get_max_threads();
        if (num_hilos > max_hilos) {
            num_hilos = max_hilos;
            cout << "=====================================================" << endl;
            cout << "Número de hilos solicitado excede el máximo, se usará: " << num_hilos << endl;
            cout << "=====================================================\n" << endl;
        } else {
            cout << "=======================" << endl;
            cout << "Usando " << num_hilos << " hilos." << endl;
            cout << "=======================\n" << endl;
        }

        ResultadoDeteccion R;
        int total_votos = votos.size();
        
        // Inicializar contadores globales
        R.anomalias_flujo_excesivo = 0;
        R.anomalias_concentracion = 0;
        R.anomalias_duplicados = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        // FASE 1: Análisis global para identificar patrones anómalos
        std::unordered_map<std::string, int> contador_dni_global;
        std::unordered_map<std::string, int> contador_region_candidato_global;
        std::unordered_map<std::string, int> contador_timestamp_global;

        // Construir contadores globales (secuencial para evitar condiciones de carrera)
        for (const auto& v : votos) {
            contador_dni_global[v.dni]++;
            
            std::string clave_conc = v.region + "|" + v.candidato;
            contador_region_candidato_global[clave_conc]++;
            
            std::string t_clave = v.timestamp.substr(0, 16); // Agrupar por minuto
            contador_timestamp_global[t_clave]++;
        }

        // Calcular umbrales estadísticamente significativos
        // Para duplicados: cualquier DNI que aparezca más de una vez es anómalo
        
        // Para concentración: usar desviación estándar como criterio
        std::vector<int> valores_concentracion;
        for (const auto& pair : contador_region_candidato_global) {
            valores_concentracion.push_back(pair.second);
        }
        
        double media_conc = 0.0, std_conc = 0.0;
        if (!valores_concentracion.empty()) {
            media_conc = std::accumulate(valores_concentracion.begin(), valores_concentracion.end(), 0.0) / valores_concentracion.size();
            
            double suma_cuadrados = 0.0;
            for (int val : valores_concentracion) {
                suma_cuadrados += (val - media_conc) * (val - media_conc);
            }
            std_conc = std::sqrt(suma_cuadrados / valores_concentracion.size());
        }
        
        // Umbral: media + 2*desviación estándar (95% de confianza estadística)
        int umbral_concentracion = static_cast<int>(media_conc + 2.0 * std_conc);
        umbral_concentracion = std::max(umbral_concentracion, 50); // Mínimo 50 para casos extremos

        // Para flujo: similar análisis estadístico
        std::vector<int> valores_flujo;
        for (const auto& pair : contador_timestamp_global) {
            valores_flujo.push_back(pair.second);
        }
        
        double media_flujo = 0.0, std_flujo = 0.0;
        if (!valores_flujo.empty()) {
            media_flujo = std::accumulate(valores_flujo.begin(), valores_flujo.end(), 0.0) / valores_flujo.size();
            
            double suma_cuadrados = 0.0;
            for (int val : valores_flujo) {
                suma_cuadrados += (val - media_flujo) * (val - media_flujo);
            }
            std_flujo = std::sqrt(suma_cuadrados / valores_flujo.size());
        }
        
        int umbral_flujo = static_cast<int>(media_flujo + 2.0 * std_flujo);
        umbral_flujo = std::max(umbral_flujo, 100); // Mínimo 100 para casos extremos

        cout << "Umbrales calculados estadísticamente:" << endl;
        cout << "- Concentración: " << umbral_concentracion << " (media: " << media_conc << ", std: " << std_conc << ")" << endl;
        cout << "- Flujo: " << umbral_flujo << " (media: " << media_flujo << ", std: " << std_flujo << ")" << endl;

        // FASE 2: Clasificación paralela usando los patrones identificados
        std::vector<std::vector<Voto>> locales_validos(num_hilos);
        std::vector<std::vector<Voto>> locales_anomalos(num_hilos);
        std::vector<int> flujo(num_hilos, 0), concentracion(num_hilos, 0), duplicados(num_hilos, 0);

        #pragma omp parallel num_threads(num_hilos)
        {
            int id = omp_get_thread_num();
            int inicio = (total_votos * id) / num_hilos;
            int fin = (total_votos * (id + 1)) / num_hilos;

            #pragma omp critical
            {
                cout << "Hilo " << id << " procesará votos desde " << inicio << " hasta " << fin - 1
                    << " (total: " << (fin - inicio) << " votos)" << endl;
            }

            for (int i = inicio; i < fin; ++i) {
                const Voto& v = votos[i];
                Voto voto = v;

                bool es_anomalo = false;
                int tipo = -1;

                // --- Anomalía 1: Duplicados por DNI (usar contadores globales) ---
                if (contador_dni_global[v.dni] > 1) {
                    es_anomalo = true;
                    tipo = 1;
                    duplicados[id]++;
                }

                // --- Anomalía 2: Concentración sospechosa (usar umbral estadístico) ---
                std::string clave_conc = v.region + "|" + v.candidato;
                if (contador_region_candidato_global[clave_conc] > umbral_concentracion) {
                    if (!es_anomalo) { // Solo cambiar tipo si no es ya anómalo
                        tipo = 2;
                    }
                    es_anomalo = true;
                    concentracion[id]++;
                }

                // --- Anomalía 3: Flujo excesivo (usar umbral estadístico) ---
                std::string t_clave = v.timestamp.substr(0, 16);
                if (contador_timestamp_global[t_clave] > umbral_flujo) {
                    if (!es_anomalo) { // Solo cambiar tipo si no es ya anómalo
                        tipo = 3;
                    }
                    es_anomalo = true;
                    flujo[id]++;
                }

                voto.anomalia_detectada = es_anomalo;
                voto.tipo_anomalia = tipo;

                if (es_anomalo) {
                    locales_anomalos[id].push_back(voto);
                } else {
                    locales_validos[id].push_back(voto);
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        R.tiempo_proceso_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // FASE 3: Consolidar resultados
        for (int i = 0; i < num_hilos; ++i) {
            R.anomalos.insert(R.anomalos.end(), locales_anomalos[i].begin(), locales_anomalos[i].end());
            R.validos.insert(R.validos.end(), locales_validos[i].begin(), locales_validos[i].end());

            R.anomalias_duplicados     += duplicados[i];
            R.anomalias_concentracion  += concentracion[i];
            R.anomalias_flujo_excesivo += flujo[i];
        }

        // FASE 4: Calcular métricas estadísticas correctas
        // VP = Verdaderos Positivos: casos que son anómalos reales Y fueron detectados como anómalos
        // FP = Falsos Positivos: casos normales que fueron detectados como anómalos
        // FN = Falsos Negativos: casos anómalos reales que no fueron detectados
        // VN = Verdaderos Negativos: casos normales que fueron detectados como normales
        
        int VP = 0, FP = 0, FN = 0, VN = 0;
        
        // Contar en votos detectados como anómalos
        for (const auto& v : R.anomalos) {
            if (v.anomalo && v.anomalia_detectada) {
                VP++; // Correctamente identificado como anómalo
            } else if (!v.anomalo && v.anomalia_detectada) {
                FP++; // Incorrectamente identificado como anómalo
            }
        }
        
        // Contar en votos detectados como válidos
        for (const auto& v : R.validos) {
            if (v.anomalo && !v.anomalia_detectada) {
                FN++; // Anómalo real no detectado
            } else if (!v.anomalo && !v.anomalia_detectada) {
                VN++; // Correctamente identificado como normal
            }
        }

        // Calcular métricas con verificación de división por cero
        R.precision = (VP + FP) > 0 ? static_cast<double>(VP) / (VP + FP) : 0.0;
        R.recall = (VP + FN) > 0 ? static_cast<double>(VP) / (VP + FN) : 0.0;
        R.f1_score = (R.precision + R.recall) > 0.0 ? 
                     2.0 * R.precision * R.recall / (R.precision + R.recall) : 0.0;

        // Información de diagnóstico
        cout << "\n=== RESULTADOS DE DETECCIÓN ===" << endl;
        cout << "Total de votos procesados: " << total_votos << endl;
        cout << "Votos válidos detectados: " << R.validos.size() << endl;
        cout << "Votos anómalos detectados: " << R.anomalos.size() << endl;
        cout << "Anomalías por duplicados: " << R.anomalias_duplicados << endl;
        cout << "Anomalías por concentración: " << R.anomalias_concentracion << endl;
        cout << "Anomalías por flujo excesivo: " << R.anomalias_flujo_excesivo << endl;
        cout << "\n=== MÉTRICAS ESTADÍSTICAS ===" << endl;
        cout << "VP (Verdaderos Positivos): " << VP << endl;
        cout << "FP (Falsos Positivos): " << FP << endl;
        cout << "FN (Falsos Negativos): " << FN << endl;
        cout << "VN (Verdaderos Negativos): " << VN << endl;
        cout << "Precisión: " << R.precision << endl;
        cout << "Recall: " << R.recall << endl;
        cout << "F1-Score: " << R.f1_score << endl;
        cout << "Tiempo de procesamiento: " << R.tiempo_proceso_ms << " ms" << endl;
        cout << "===============================" << endl;

        return R;
    }

}
