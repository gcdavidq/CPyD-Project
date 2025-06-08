/*
Script que incluye todas las librerías necesarias para la simulación de un nodo esclavo en un sistema de votación.
*/
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include <queue>
#include <atomic>

#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/common/config.hpp"
#include "VOTACION/simulacion/simulacion_llegada.hpp"
#include "VOTACION/deteccion/detectar_anomalias.hpp"
#include "VOTACION/protocolo/protocolo.hpp"
#include "VOTACION/procesamiento/procesar_lote.hpp"

using namespace std;

// Función que ejecuta el nodo esclavo
void ejecutarNodoEsclavo(int nodo_id, int nodo_maestro) {
    cout << "Iniciando nodo esclavo " << nodo_id << endl;
    
    // Detectar capacidades del nodo
    CapacidadNodo capacidad = detectarCapacidadNodo();
    cout << "Nodo " << nodo_id << " capacidad: " << capacidad.num_hilos 
         << " hilos, GPU: " << (capacidad.tiene_gpu ? "Sí" : "No") << endl;
    
    // Informar capacidades al nodo maestro
    vector<char> buffer(sizeof(CapacidadNodo));
    memcpy(buffer.data(), &capacidad, sizeof(CapacidadNodo));
    MPI_Send(buffer.data(), buffer.size(), MPI_CHAR, nodo_maestro, TAG_CAPACIDAD_NODO, MPI_COMM_WORLD);
    
    // Cargar datos de votación para este nodo
    string archivo_region = ARCHIVO_ENTRADA_BASE + to_string(nodo_id) + ".csv";
    vector<Voto> votos_recibidos;
    
    // Lanzar un hilo para simular la llegada de votos en tiempo real
    thread hilo_simulacion(simularLlegadaVotos, archivo_region, nodo_id, ref(votos_recibidos));
    
    // Estadísticas locales
    Estadisticas estadisticas_locales;
    
    // Variables para seguimiento de rendimiento
    auto tiempo_inicio = chrono::system_clock::now();
    auto ultimo_reporte_rendimiento = tiempo_inicio;
    float tiempo_promedio_lote = 0.0f;
    int lotes_procesados = 0;
    
    // Mutex para proteger el acceso a los votos recibidos
    mutex mtx_votos;
    
    // Flag para indicar finalización
    bool continuar_procesando = true;
    
    // Bucle principal de procesamiento
    while (continuar_procesando) {
        // Verificar si hay nuevos votos para procesar
        vector<Voto> lote_votos;
        {
            lock_guard<mutex> lock(mtx_votos);
            if (!votos_recibidos.empty()) {
                // Tomar un máximo de 1000 votos para procesar
                int votos_a_tomar = min(1000, static_cast<int>(votos_recibidos.size()));
                lote_votos.insert(lote_votos.end(), votos_recibidos.begin(), 
                                 votos_recibidos.begin() + votos_a_tomar);
                votos_recibidos.erase(votos_recibidos.begin(), 
                                     votos_recibidos.begin() + votos_a_tomar);
            }
        }
        
        if (!lote_votos.empty()) {
            // Crear un lote de trabajo
            LoteTrabajo lote;
            lote.id_lote = lotes_procesados;
            lote.votos = lote_votos;
            lote.inicio_timestamp = lote.votos.front().timestamp;
            lote.fin_timestamp = lote.votos.back().timestamp;
            
            // Procesar el lote
            auto inicio_lote = chrono::system_clock::now();
            Estadisticas stats_lote = procesarLote(lote, capacidad.tiene_gpu);
            auto fin_lote = chrono::system_clock::now();
            
            // Actualizar tiempo promedio
            chrono::duration<float> duracion_lote = fin_lote - inicio_lote;
            if (lotes_procesados == 0) {
                tiempo_promedio_lote = duracion_lote.count();
            } else {
                tiempo_promedio_lote = (tiempo_promedio_lote * lotes_procesados + duracion_lote.count()) / 
                                      (lotes_procesados + 1);
            }
            
            // Actualizar estadísticas locales
            estadisticas_locales.combinar(stats_lote);
            lotes_procesados++;
            
            // Serializar estadísticas del lote
            vector<char> buffer_stats;
            serializarEstadisticas(stats_lote, buffer_stats);
            
            // Enviar resultados al nodo maestro
            MPI_Send(buffer_stats.data(), buffer_stats.size(), MPI_CHAR, nodo_maestro, TAG_REPORTE_STATS, MPI_COMM_WORLD);
            
            cout << "Nodo " << nodo_id << " completó lote " << lote.id_lote << " con " 
                 << lote.votos.size() << " votos en " << duracion_lote.count() << " segundos" << endl;
        }
        
        // Reportar rendimiento periódicamente
        auto ahora = chrono::system_clock::now();
        chrono::duration<float> tiempo_desde_reporte = ahora - ultimo_reporte_rendimiento;
        
        if (tiempo_desde_reporte.count() >= INTERVALO_CHEQUEO_BALANCEO / 2) {
            RendimientoNodo ren;
            ren.nodo_id = nodo_id;
            ren.tiempo_promedio_lote = tiempo_promedio_lote;
            ren.lotes_completados = lotes_procesados;
            ren.tiene_gpu = capacidad.tiene_gpu;
            
            // Calcular carga actual (ejemplo simplificado)
            chrono::duration<float> tiempo_total = ahora - tiempo_inicio;
            ren.carga_actual = (tiempo_promedio_lote > 0) ? 
                min(100.0f, (lotes_procesados * tiempo_promedio_lote * 100.0f) / tiempo_total.count()) : 0.0f;
            
            // Enviar información de rendimiento al maestro
            vector<char> buffer_ren(sizeof(RendimientoNodo));
            memcpy(buffer_ren.data(), &ren, sizeof(RendimientoNodo));
            MPI_Send(buffer_ren.data(), buffer_ren.size(), MPI_CHAR, nodo_maestro, TAG_RENDIMIENTO_NODO, MPI_COMM_WORLD);
            
            ultimo_reporte_rendimiento = ahora;
        }
        
        // Comprobar mensajes del nodo maestro para balanceo de carga
        MPI_Status estado;
        int flag = 0;
        MPI_Iprobe(nodo_maestro, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &estado);
        
        if (flag) {
            int tag = estado.MPI_TAG;
            
            switch (tag) {
                case TAG_BALANCE_CARGA: {
                    // Recibir solicitud de balanceo
                    MPI_Recv(nullptr, 0, MPI_CHAR, nodo_maestro, TAG_BALANCE_CARGA, MPI_COMM_WORLD, &estado);
                    
                    // Tomar algunos votos pendientes para redistribuir
                    vector<Voto> votos_redistribuir;
                    {
                        lock_guard<mutex> lock(mtx_votos);
                        if (votos_recibidos.size() > 500) {  // Solo redistribuir si tenemos suficientes
                            int votos_a_tomar = min(500, static_cast<int>(votos_recibidos.size()) / 4);
                            votos_redistribuir.insert(votos_redistribuir.end(), 
                                                     votos_recibidos.end() - votos_a_tomar, 
                                                     votos_recibidos.end());
                            votos_recibidos.erase(votos_recibidos.end() - votos_a_tomar, 
                                                 votos_recibidos.end());
                        }
                    }
                    
                    if (!votos_redistribuir.empty()) {
                        // Crear un lote de trabajo para redistribuir
                        LoteTrabajo lote;
                        lote.id_lote = -1;  // ID especial para lotes redistribuidos
                        lote.votos = votos_redistribuir;
                        lote.inicio_timestamp = lote.votos.front().timestamp;
                        lote.fin_timestamp = lote.votos.back().timestamp;
                        
                        // Serializar lote
                        vector<char> buffer_lote;
                        serializarLote(lote, buffer_lote);
                        
                        // Enviar lote al maestro para redistribución
                        MPI_Send(buffer_lote.data(), buffer_lote.size(), MPI_CHAR, 
                                nodo_maestro, TAG_ENVIO_TRABAJO, MPI_COMM_WORLD);
                        
                        cout << "Nodo " << nodo_id << " redistribuyó " << votos_redistribuir.size() 
                             << " votos para balanceo de carga" << endl;
                    } else {
                        // Informar que no hay trabajo para redistribuir
                        MPI_Send(nullptr, 0, MPI_CHAR, nodo_maestro, TAG_SIN_TRABAJO, MPI_COMM_WORLD);
                    }
                    
                    break;
                }
                
                case TAG_ENVIO_TRABAJO: {
                    // Recibir trabajo adicional (desde balanceo)
                    int tamanio_buffer;
                    MPI_Get_count(&estado, MPI_CHAR, &tamanio_buffer);
                    vector<char> buffer(tamanio_buffer);
                    
                    MPI_Recv(buffer.data(), buffer.size(), MPI_CHAR, nodo_maestro, 
                            TAG_ENVIO_TRABAJO, MPI_COMM_WORLD, &estado);
                    
                    // Deserializar lote
                    LoteTrabajo lote = deserializarLote(buffer);
                    
                    // Añadir votos a la cola de procesamiento
                    {
                        lock_guard<mutex> lock(mtx_votos);
                        votos_recibidos.insert(votos_recibidos.end(), 
                                              lote.votos.begin(), lote.votos.end());
                    }
                    
                    cout << "Nodo " << nodo_id << " recibió " << lote.votos.size() 
                         << " votos adicionales para procesar" << endl;
                    
                    break;
                }
                
                case TAG_FINALIZAR: {
                    // Recibir señal de finalización
                    MPI_Recv(nullptr, 0, MPI_CHAR, nodo_maestro, TAG_FINALIZAR, MPI_COMM_WORLD, &estado);
                    continuar_procesando = false;
                    
                    cout << "Nodo " << nodo_id << " recibió señal de finalización" << endl;
                    
                    break;
                }
            }
        }
        
        // Si no hay votos para procesar, dormir un poco
        if (lote_votos.empty() && flag == 0) {
            this_thread::sleep_for(chrono::milliseconds(100));
        }
    }
    
    // Enviar señal de finalización y reporte final
    MPI_Send(nullptr, 0, MPI_CHAR, nodo_maestro, TAG_RESULTADO_FINAL, MPI_COMM_WORLD);
    
    // Esperar a que termine el hilo de simulación
    if (hilo_simulacion.joinable()) {
        hilo_simulacion.join();
    }
    
    cout << "Nodo " << nodo_id << " completó su ejecución" << endl;
}