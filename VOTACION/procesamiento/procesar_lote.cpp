#include "VOTACION/procesamiento/procesar_lote.hpp"
#include "VOTACION/deteccion/detectar_anomalias.hpp"
#include <iostream>
#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/common/config.hpp"
#include <vector>
#include <iomanip>

using namespace std;
using namespace deteccion;

// Enumeración para tipos de anomalías
enum TipoAnomalia {
    NINGUNA = -1,
    CONCENTRACION_CANDIDATO = 2,
    DNI_DUPLICADO = 1,
    FLUJO_EXCESIVO = 3,
};

std::string anomaliaToString(TipoAnomalia tipo) {
    switch(tipo) {
        case DNI_DUPLICADO: return "DNI_DUPLICADO";
        case CONCENTRACION_CANDIDATO: return "CONCENTRACION_CANDIDATO";
        case FLUJO_EXCESIVO: return "FLUJO_EXCESIVO";
        default: return "NINGUNA"; 
    }
}
/*
void mostrarVotosValidos(const ResultadoDeteccion& resultado) {
    std::cout << "\n=== VOTOS VÁLIDOS (" << resultado.validos.size() << ") ===" << std::endl;
    for (const auto& voto : resultado.validos) {
        std::cout << "[" << voto.timestamp << "] "
                  << "Reg: " << std::setw(3) << voto.region
                  << " | DNI: " << std::setw(3) << voto.dni
                  << " | Candidato: " << std::setw(3) << voto.candidato
                  << " | Anómalo: " << (voto.anomalo ? "Sí" : "No")
                  << " | Detectado: " << (voto.anomalia_detectada ? "Sí" : "No")
                  << " | Tipo: " << anomaliaToString(static_cast<TipoAnomalia>(voto.tipo_anomalia)) << "\n";
    }
}

void mostrarVotosAnomalos(const ResultadoDeteccion& resultado) {
    std::cout << "\n=== VOTOS ANÓMALOS (" << resultado.anomalos.size() << ") ===" << std::endl;
    for (const auto& voto : resultado.anomalos) {
        std::cout << "[" << voto.timestamp << "] "
                  << "Reg: " << std::setw(3) << voto.region
                  << " | DNI: " << std::setw(3) << voto.dni
                  << " | Candidato: " << std::setw(3) << voto.candidato
                  << " | Anómalo: " << (voto.anomalo ? "Sí" : "No")
                  << " | Detectado: " << (voto.anomalia_detectada ? "Sí" : "No")
                  << " | Tipo: " << anomaliaToString(static_cast<TipoAnomalia>(voto.tipo_anomalia)) << "\n";
    }
}
*/
Estadisticas procesarLote(LoteTrabajo& lote, bool tiene_gpu) {
    std::vector<Voto>lote_votos=lote.votos;
    int total_anom3=0;
    for (const auto& voto : lote_votos) {
        if (voto.anomalo) { total_anom3++;
                        
                }
        }
    std::cout << "[DEBUG] Votos anomalos antes de pasar a detectar anomalias CPU O GPU"<<total_anom3 << std::endl;
    
    
    Estadisticas stats;
    if (lote.votos.empty()) return stats;
    
    

    deteccion::ResultadoDeteccion resultado;

    if (tiene_gpu) {
#ifdef USE_CUDA
        resultado = deteccion::detectarAnomaliasCUDA(lote.votos);
        lote.votos = resultado.validos;
        lote.votos.insert(lote.votos.end(), resultado.anomalos.begin(), resultado.anomalos.end());
#else
        resultado = deteccion::detectarAnomaliasCPU(lote.votos, NUM_HILOS_PARA_ALG);
        lote.votos = resultado.validos;
        lote.votos.insert(lote.votos.end(), resultado.anomalos.begin(), resultado.anomalos.end());
#endif
    } else {
        //DETECTAR ANOMALIA CPU 
        resultado = deteccion::detectarAnomaliasCPU(lote.votos, NUM_HILOS_PARA_ALG);
        //mostrarVotosValidos(resultado);
        //mostrarVotosAnomalos(resultado);

        lote.votos = resultado.validos;
        lote.votos.insert(lote.votos.end(), resultado.anomalos.begin(), resultado.anomalos.end());
        

        // Mostrar estadísticas
        cout << "\n--- RESULTADO DE DETECCIÓN POR LOTE ---" << endl;
        cout << "Votos válidos          : " << resultado.validos.size() << endl;
        cout << "Votos anómalos detect. : " << resultado.anomalos.size() << endl;
        cout << "Tiempo de ejecución    : " << resultado.tiempo_proceso_ms << " ms\n" << endl;
        
        cout << "Anomalías detectadas por tipo:" << endl;
        cout << "  → Flujo excesivo         : " << resultado.anomalias_flujo_excesivo << endl;
        cout << "  → Concentración candidato: " << resultado.anomalias_concentracion << endl;
        cout << "  → DNIs duplicados        : " << resultado.anomalias_duplicados << endl;
        
        cout << "\nMétricas de evaluación:" << endl;
        cout << "  • Precisión: " << resultado.precision << endl;
        cout << "  • Recall   : " << resultado.recall << endl;
        cout << "  • F1 Score : " << resultado.f1_score << endl;
        
        
    }

    stats.total_votos = lote.votos.size();
    for (const auto& voto : lote.votos) {
        stats.votos_por_region[voto.region]++;
        stats.votos_por_candidato[voto.candidato]++;
        stats.votos_por_candidato_por_region[voto.region][voto.candidato]++;

        if (voto.anomalo) {
            stats.anomalias_reales++;
            if (voto.anomalia_detectada) {
                stats.anomalias_detectadas++;
                stats.anomalias_detectadas_por_region[voto.region][voto.candidato]++;
                stats.anomalias_detectadas_por_candidato[voto.candidato][voto.region]++;
            } else {
                stats.falsos_negativos++;
            }
        } else if (voto.anomalia_detectada) {
            stats.falsos_positivos++;
            stats.anomalias_detectadas++;
            stats.anomalias_detectadas_por_region[voto.region][voto.candidato]++;
            stats.anomalias_detectadas_por_candidato[voto.candidato][voto.region]++;
        }
    }

    return stats;
}
