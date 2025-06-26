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
#include <sys/resource.h>
#include <sys/time.h>

#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/common/config.hpp"
#include "VOTACION/simulacion/simulacion_llegada.hpp"
#include "VOTACION/deteccion/detectar_anomalias.hpp"
#include "VOTACION/protocolo/protocolo.hpp"
#include "VOTACION/procesamiento/procesar_lote.hpp"
#include "VOTACION/estadisticas/estadisticas.hpp"
#include "VOTACION/rendimiento/rendimiento.hpp"

using namespace std;

// Función que ejecuta el nodo esclavo
void ejecutarNodoEsclavo(int nodo_id, int nodo_maestro) {
    cout << "Iniciando nodo esclavo " << nodo_id << endl;
    
    // Detectar capacidades del nodo
    CapacidadNodo capacidad = detectarCapacidadNodo();
    cout << "Nodo " << nodo_id << " capacidad: " << capacidad.num_hilos 
         << " hilos, GPU: " << (capacidad.tiene_gpu ? "Sí" : "No") << endl;
    
    // Informar capacidades al nodo maestro
    //Creamos un vector llamado buffer que almacena las capacidades del nodo (en bytes)
    vector<char> buffer(sizeof(CapacidadNodo)); 
    //copiamos los bytes del objeto capacidad al buffer
    memcpy(buffer.data(), &capacidad, sizeof(CapacidadNodo));
    //Enviamos al nodo maestro
    MPI_Send(
        buffer.data(), 
        buffer.size(), 
        MPI_CHAR, nodo_maestro, 
        TAG_CAPACIDAD_NODO, 
        MPI_COMM_WORLD);
    
    // Cargar datos de votación para este nodo
    string archivo_region = ARCHIVO_ENTRADA_BASE + to_string(nodo_id) + ".csv";
    vector<Voto> votos_recibidos;
    

    mutex mtx_votos;
    atomic<bool> simulacion_terminada(false);


    // Lanzar un hilo para simular la llegada de votos en tiempo real
    thread hilo_simulacion([&]() {
            simularLlegadaVotos(archivo_region, nodo_id, votos_recibidos,mtx_votos);
            simulacion_terminada = true;
    });

    

        
    /*
    thread hilo_simulacion(simularLlegadaVotos, archivo_region, nodo_id, ref(votos_recibidos));
    */
    // Estadísticas locales

    Estadisticas estadisticas_locales;
    
    // Variables para seguimiento de rendimiento
    auto tiempo_inicio = chrono::system_clock::now();
    auto ultimo_reporte_rendimiento = tiempo_inicio;
    float tiempo_promedio_lote = 0.0f;
    double tiempo_promedio_comunicacion =0;
    int lotes_procesados = 0;
    int tiempo_total_lotes = 0.0f;

    
    // Mutex para proteger el acceso a los votos recibidos
    
    bool ya_notificado_desocupado = false;
    
    // Flag para indicar finalización
    bool continuar_procesando = true;
    
    // Bucle principal de procesamiento
while (continuar_procesando) {
    // Espera artificial para permitir que se acumulen más votos
    
    cout<<"votos del nodo: "<< nodo_id<<" que esperan ser procesados: "<< votos_recibidos.size()<<endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(6000));

    
    // Verificar si hay nuevos votos para procesar
    vector<Voto> lote_votos;
    {
        lock_guard<mutex> lock(mtx_votos);
        int total_anom = 0;
        for (const auto& v : votos_recibidos) {
            if (v.anomalo) total_anom++;
        }
        cout << "[DEBUG] Total acumulado en votos_recibidos: " << total_anom << " votos anómalos en el nodo: "<<nodo_id<<endl;
        
        
        if (!votos_recibidos.empty()) {
            
            //PROBAR LA ANOMALIA DE LOS VOTOS
            int total_anom1=0;

            for (const auto& voto : votos_recibidos) {
                if (voto.anomalo) { 
                    total_anom1++;
                }
            }
            cout << "[DEBUG] Total acumulado de votos anomalos recibidos"<<total_anom1 << "del nodo: "<<nodo_id<< endl; 

            // Tomar un máximo de 1000 votos para procesar
            int votos_a_tomar = min(TAM_LOTE_POR_DEFECTO, static_cast<int>(votos_recibidos.size()));
            lote_votos.insert(lote_votos.end(), votos_recibidos.begin(), 
                             votos_recibidos.begin() + votos_a_tomar);
            votos_recibidos.erase(votos_recibidos.begin(), 
                                 votos_recibidos.begin() + votos_a_tomar);
        }
    }
    
    // * NUEVA LÓGICA DE TERMINACIÓN *
    // Verificar si el nodo ha completado todo su trabajo
    bool trabajo_local_terminado = false;
    {
        lock_guard<mutex> lock(mtx_votos);
        trabajo_local_terminado = votos_recibidos.empty() && simulacion_terminada.load();
    }
    
    if (trabajo_local_terminado && lote_votos.empty()) {
        if (!ya_notificado_desocupado) {
            // Notificar al maestro que está desocupado
            MPI_Send(nullptr, 0, MPI_CHAR, nodo_maestro, TAG_NODO_DESOCUPADO, MPI_COMM_WORLD);
            ya_notificado_desocupado = true;
            cout << "[DEBUG] Nodo " << nodo_id << " notificó que está desocupado" << endl;
        }
        
        // Esperar un poco por si llega trabajo de balanceo de carga
        // Si después de un tiempo razonable no llega nada, terminar
        auto tiempo_espera_inicio = chrono::system_clock::now();
        bool trabajo_recibido = false;
        const int tiempo_espera_max = 100;
        
        while (chrono::duration_cast<chrono::seconds>(
                chrono::system_clock::now() - tiempo_espera_inicio).count() < tiempo_espera_max) {
            
            // Comprobar mensajes del maestro
            MPI_Status estado;
            int flag = 0;
            MPI_Iprobe(nodo_maestro, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &estado);
            
            if (flag) {
                int tag = estado.MPI_TAG;
                
                if (tag == TAG_ENVIO_TRABAJO) {
                    // Recibió trabajo nuevo, salir del bucle de espera
                    trabajo_recibido = true;
                    break;

                } else if (tag == TAG_FINALIZAR) {
                    // Recibió orden de finalización
                    MPI_Recv(nullptr, 0, MPI_CHAR, nodo_maestro, TAG_FINALIZAR, MPI_COMM_WORLD, &estado);
                    continuar_procesando = false;
                    cout << "Nodo " << nodo_id << " recibió señal de finalización" << endl;
                    break;

                } else {
                    cout << "[DEBUG] Nodo " << nodo_id << " procesando mensaje con tag " << tag << endl;
                    break;
                }
            }
 
            // Dormir un poco antes de verificar de nuevo
            this_thread::sleep_for(chrono::milliseconds(100));
        }

        // Verificar nuevamente si llegaron votos durante la espera
        {
            lock_guard<mutex> lock(mtx_votos);
            if (!votos_recibidos.empty()) {
                trabajo_recibido = true;
                cout << "[DEBUG] Nodo " << nodo_id << " encontró nuevos votos durante espera" << endl;
                break;
            }
        }
        
        // Si no recibió trabajo después del tiempo de espera, asumir que puede terminar
        if (!trabajo_recibido && continuar_procesando) {
            cout << "[DEBUG] Nodo " << nodo_id << " terminando por falta de trabajo" << endl;
            continuar_procesando = false;
            break;
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
        tiempo_total_lotes= tiempo_total_lotes + duracion_lote.count();
        
        // Actualizar estadísticas locales
        estadisticas_locales.combinar(stats_lote);
        lotes_procesados++;
        
        // Serializar estadísticas del lote
        vector<char> buffer_stats;
        serializarEstadisticas(stats_lote, buffer_stats);
        
        // Enviar resultados al nodo maestro
        double inicio_com = MPI_Wtime();
        MPI_Send(buffer_stats.data(), buffer_stats.size(), MPI_CHAR, nodo_maestro, TAG_REPORTE_STATS, MPI_COMM_WORLD);
        double fin_com = MPI_Wtime();
        double tiempo_envio= fin_com - inicio_com;
        cout <<"TIEMPO DE DEMORA EN ENVIÓ: "<<tiempo_envio;
        if (lotes_procesados == 0) {
            tiempo_promedio_comunicacion = tiempo_envio;
        } else {
            tiempo_promedio_comunicacion = ((tiempo_promedio_comunicacion * (lotes_procesados - 1)) + tiempo_envio) 
            / lotes_procesados;
        }

        cout<<"TIEMPO PROMEDIO DE ENVIO DE DATOS AL NODO MAESTRO: "<<tiempo_promedio_comunicacion<< "del nodo: "<<nodo_id<<endl;


        
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
        ren.tiempo_comunicacion_mpi = tiempo_promedio_comunicacion;
        ren.num_hilos=capacidad.num_hilos;
        
        // Calcular carga actual 
        chrono::duration<float> tiempo_total = ahora - tiempo_inicio;
        float carga = obtenerUsoCPU(tiempo_total.count());
        float hilos = omp_get_max_threads();
        cout<<"NODO "<<nodo_id << "Uso real de CPU: " << carga << " %" << "con" << hilos<< "hilos" << endl;
        /*

        //Carga basada en votos pendientes y capacidad 
        float votos_pendientes = votos_recibidos.size();
        float carga_por_cola = min(100.0f, (votos_pendientes/static_cast<float>(TAM_LOTE_POR_DEFECTO)) * 100.0f);

        //Carga basada en utilización de tiempos
        float tiempo_ocupado = lotes_procesados * tiempo_promedio_lote;
        float tiempo_transcurrido = tiempo_total.count();
        float carga_por_tiempo = (tiempo_transcurrido>0)?
                min(100.0f, (tiempo_ocupado/tiempo_transcurrido)*100.0f):0.0f;
        //Combinar ambas métricas */
        ren.carga_actual = carga;
        //saber que metrica fue escogida
        /*
        if (carga_por_cola>carga_por_tiempo){
            cout<<"Carga por cola escogida por tener mayor carga"<<endl;
        } else{
            cout<<"Carga por tiempo escogida por tener mayor carga"<<endl;
        }
        cout << "   Carga actual del nodo: " << nodo_id << " es: " << ren.carga_actual<<" %"<< endl;
        */

        // Enviar información de rendimiento al maestro
        //imprimirInfoNodoWeb(nodo_id, ren);
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
                    if (votos_recibidos.size() > 100) {  // Solo redistribuir si tenemos suficientes
                        int votos_a_tomar = min(TAM_LOTE_POR_DEFECTO, static_cast<int>(votos_recibidos.size()) / 4);
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
                    cout <<"CANTIDAD DE VOTOS A REDISTRIBUIR: "<< votos_redistribuir.size()<<endl;
                    
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
    
    // Si no hay votos para procesar y no hay mensajes, dormir un poco
    if (lote_votos.empty() && flag == 0 && continuar_procesando) {
        //cout<<"[DEBUG] Hilo del nodo " << nodo_id<<" DESCANSANDO"<<endl;
        this_thread::sleep_for(chrono::milliseconds(100));
    }

    
}

// Asegurar que se envía la señal de finalización
if (continuar_procesando == false) {
    cout << "[DEBUG] Nodo " << nodo_id << " enviando señal de resultado final" << endl;
    MPI_Send(nullptr, 0, MPI_CHAR, nodo_maestro, TAG_RESULTADO_FINAL, MPI_COMM_WORLD);
}


// Esperar a que termine el hilo de simulación con timeout
if (hilo_simulacion.joinable()) {
    cout << "[DEBUG] Nodo " << nodo_id << " esperando que termine hilo de simulación" << endl;
    
    // Intentar join con timeout implícito
    auto tiempo_join_inicio = chrono::system_clock::now();
    while (hilo_simulacion.joinable()) {
        auto tiempo_transcurrido = chrono::duration_cast<chrono::seconds>(
            chrono::system_clock::now() - tiempo_join_inicio).count();
            
        if (tiempo_transcurrido > 100) {
            cout << "[WARNING] Nodo " << nodo_id << " forzando finalización del hilo de simulación" << endl;
            // En C++, no podemos forzar detach/join, pero podemos intentar detach
            hilo_simulacion.detach();
            break;
        }
        
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    
    if (hilo_simulacion.joinable()) {
        hilo_simulacion.join();
    }
}
//TIEMPO DE EJECUCIÓN POR NODO
chrono::duration<double> tiempo_final = chrono::system_clock::now()-tiempo_inicio;


cout << "Nodo " << nodo_id << " completó su ejecución en: " << tiempo_final.count()<<"segundos"<< endl;
cout <<"Nodo" << nodo_id << "demoró solo ejecutando lotes "<< tiempo_total_lotes << "segundos"<<endl;
cout<<"Nodo"<<nodo_id<<"tiempo promedio por lote"<<tiempo_promedio_lote<<"Segundos"<<endl;
cout << "Nodo" << nodo_id <<"usó: "<<  NUM_HILOS_PARA_ALG<<"para el algoritmo de votos"<<endl;
cout << "Lotes completados para el nodo: " << nodo_id << " : " << lotes_procesados <<"lotes"<< endl;
imprimirEstadisticas(estadisticas_locales, nodo_id);
}