/*
Script donde se encuentran las funciones del nodo maestro
*/
#include <iostream>
#include <vector>
#include <thread>
#include <omp.h>
#include <iomanip>
#include<mpi.h>
#include <queue>
#include <set>
#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/common/config.hpp"
#include "VOTACION/balanceo/estado_balanceo.hpp"
#include "VOTACION/protocolo/protocolo.hpp"
#include "VOTACION/balanceo/balanceo_carga.hpp"
#include "VOTACION/estadisticas/estadisticas.hpp"
#include <curl/curl.h>
#include <json/json.h>
#include "VOTACION/estadisticas/web_stats_sender.hpp"



using namespace std; 


void ejecutarNodoMaestro(int num_nodos){
    WebStatsSender web_sender("http://localhost:5000");

    int esclavos_finalizados = 0;
    int esclavos_desocupados= 0; 

    cout<<"Iniciando nodo maestro con "<<num_nodos-1<< " nodos esclavos" <<endl;

    //Creamos objeto tipo CapacidadNodo para detectar capacidad de nodo maestro
    CapacidadNodo capacidad_maestro = detectarCapacidadNodo();

    cout << "Capacidad del maestro: "<<capacidad_maestro.num_hilos << " hilos, GPU: "
        << (capacidad_maestro.tiene_gpu ? " Si" : " NO") <<endl;

    //Estructura para gestionar los nodos
    vector <CapacidadNodo> capacidades_nodos (num_nodos); //creamos un vector de capacidades
    capacidades_nodos[MAESTRO] = capacidad_maestro;
    
    vector<RendimientoNodo> rendimiento_nodos(num_nodos); //creamos un vector de rendimiento
    for (int i =0; i<num_nodos; i++){
        web_sender.enviarInfoNodo(i, rendimiento_nodos[i]);
        rendimiento_nodos[i].nodo_id = i; //Asignamos el id del nodo
        rendimiento_nodos[i].tiempo_promedio_lote = 0.0f; //inicializamos el tiempo promedio
        rendimiento_nodos[i].lotes_completados = 0; //inicializamos los lotes completados
        rendimiento_nodos[i].lotes_asignados = 0; //inicializamos los lotes asignados
        rendimiento_nodos[i].tiene_gpu = false; //inicializamos la GPU
        rendimiento_nodos[i].carga_actual = 0.0f; //inicializamos la carga actual
        rendimiento_nodos[i].ultimo_reporte = chrono::system_clock::now(); //inicializamos el tiempo de reporte
        rendimiento_nodos[i].tiempo_comunicacion_mpi=0;
        
    }

    //Recibir informacion sobre capacidades de los nodos
    for (int i =1; i<num_nodos; i++){

        MPI_Status estado; //MPI_Status es una estructura que contiene información sobre el estado de la comunicación
        //sizeof operador que se usa para obtener el tamaño en bytes de un tipo de dato o variable
        //size_t tipo de dato (el tipo de resultado de sizeof)
        //Creamos un buffer para recibir la información de los nodos, que tiene el tamaño de la estructura CapacidadNodo 
        vector<char> buffer(sizeof(CapacidadNodo));    
        
        MPI_Recv(
            buffer.data(), //Datos a recibir
            buffer.size(), //Tamaño del buffer
            MPI_CHAR, //Tipo de dato
            i, //ID del nodo que envía
            TAG_CAPACIDAD_NODO, //Etiqueta del mensaje
            MPI_COMM_WORLD, //Comunicador
            &estado //Estado de la comunicación
        ); 

        //Desearilizar capacidad
        memcpy(&capacidades_nodos[i], 
            buffer.data(), 
            sizeof(CapacidadNodo)
        ); //Copiamos los datos del buffer a la estructura de capacidad

        //Actualizar información de GPU en el seguimiento de rendimiento
        rendimiento_nodos[i].tiene_gpu= capacidades_nodos[i].tiene_gpu; //Asignamos la GPU del nodo a la estructura de rendimiento
        cout <<"Nodo "<<i<<" reporta: "<<capacidades_nodos[i].num_hilos<< " hilos, GPU: "
            << (capacidades_nodos[i].tiene_gpu ? string(capacidades_nodos[i].gpu_modelo) : "NO");

    }
    //Estadísticas globales
    Estadisticas estadisticas_globales; //Creamos una estructura para las estadísticas globales
    
    //Tiempo de inicio de la simulación
    auto tiempo_inicio = chrono::system_clock::now();
    auto ultimo_reporte = tiempo_inicio; //Inicializamos el último reporte
    auto ultimo_balanceo = tiempo_inicio; //Inicializamos el último balanceo

    //Cola de trabajo pendiente y distribución inicial
    queue<LoteTrabajo> cola_trabajo;
    int lotes_completados = 0; 
    set<int> nodos_desocupados;


    //Contador de lotes completados
    //int total_lotes = 0; 
    bool procesamiento_finalizado = false; //Variable para saber si el procesamiento ha finalizado
    bool todos_nodos_notificaron_fin = false;
    auto tiempo_ultima_actividad = chrono::system_clock::now();
    const int TIMEOUT_FINALIZACION = 30; 

    cout << "→ Iniciando bucle principal del maestro..." << endl;


    //Bucle principal de recepcion de mensajes
    while (!procesamiento_finalizado ){
        auto ahora = chrono::system_clock::now();

        // Timeout de seguridad: si no hay actividad por mucho tiempo, forzar finalización
        chrono::duration<double> tiempo_sin_actividad = ahora - tiempo_ultima_actividad;
        if (tiempo_sin_actividad.count() > TIMEOUT_FINALIZACION && esclavos_desocupados >= (num_nodos - 1)) {
            cout << "→ Timeout alcanzado y todos los nodos están desocupados. Finalizando..." << endl;
            break;
        }


        //Comprobar si es momento de realizar un reporte
        chrono::duration<double> tiempo_desde_ultimo_reporte = ahora - ultimo_reporte;
        if (tiempo_desde_ultimo_reporte.count() >= INTERVALO_REPORTE * SEGUNDOS_POR_MINUTO){
            imprimirEstadisticasWeb(estadisticas_globales, MAESTRO, web_sender); 
            ultimo_reporte = ahora;
        }

        //Comprobar si es momento de balancear carga
        chrono::duration<double> tiempo_desde_ultimo_balanceo = ahora - ultimo_balanceo;

        if (tiempo_desde_ultimo_balanceo.count() >= INTERVALO_CHEQUEO_BALANCEO){
            cout << "→ Ejecutando balanceo de carga..." << endl;
            balanceoCarga(rendimiento_nodos, cola_trabajo, num_nodos, nodos_desocupados);
            ultimo_balanceo=ahora;
        }

        // Sondear mensajes entrantes
        MPI_Status estado;
        int flag = 0;
        //Verificamos si hay un mensaje esperando a ser recibido (De manera asincrona)
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &estado);
        //Flag se vuelve 1 si hay trabajo 
        //Estado contiene el remitente y tag del mensaje
        
        if (flag) {

            tiempo_ultima_actividad = ahora;// Actualizar tiempo de actividad

            int nodo_origen = estado.MPI_SOURCE;
            int tag = estado.MPI_TAG;
            int tamanio_buffer;
            //Obtenemos cuandos elemenos fueron recibidos (bytes)
            MPI_Get_count(&estado, MPI_CHAR, &tamanio_buffer);
            
            switch (tag) {
                case TAG_REPORTE_STATS: {
                    // Recibir estadísticas de un nodo
                    vector<char> buffer(tamanio_buffer);
                    double tiempo_inicio = MPI_Wtime();
                    MPI_Recv(buffer.data(), buffer.size(), MPI_CHAR, nodo_origen, TAG_REPORTE_STATS, MPI_COMM_WORLD, &estado);
                    double tiempo_fin = MPI_Wtime();
                    double tiempo_recepcion = tiempo_fin - tiempo_inicio;
                    // Deserializar estadísticas
                    Estadisticas stats_nodo = deserializarEstadisticas(buffer);
                    
                    // Actualizar estadísticas globales
                    estadisticas_globales.combinar(stats_nodo);
                    
                    cout << "Recibido reporte del nodo " << nodo_origen << ": " 
                         << stats_nodo.total_votos << " votos procesados" << endl;
                    
                    lotes_completados++;
                    rendimiento_nodos[nodo_origen].lotes_completados++;
                    
                    break;
                }
                
                case TAG_SOLICITUD_TRABAJO: {
                    // Recibir solicitud de trabajo
                    //Null porque no se envian datos
                    MPI_Recv(nullptr, 0, MPI_CHAR, nodo_origen, TAG_SOLICITUD_TRABAJO, MPI_COMM_WORLD, &estado);
                    
                    if (!cola_trabajo.empty()) {
                        // Obtener el primer elemento (Más antiguo)
                        LoteTrabajo lote = cola_trabajo.front();
                        //lo eliminamos de la cola
                        cola_trabajo.pop();
                        
                        // Serializar lote
                        vector<char> buffer;
                        serializarLote(lote, buffer);
                        
                        // Enviar lote al nodo
                        MPI_Send(buffer.data(), buffer.size(), MPI_CHAR, nodo_origen, TAG_ENVIO_TRABAJO, MPI_COMM_WORLD);
                        
                        // Actualizar seguimiento
                        rendimiento_nodos[nodo_origen].lotes_asignados++;
                        
                        cout << "Enviado lote " << lote.id_lote << " al nodo " << nodo_origen << endl;
                    } else if (procesamiento_finalizado) {
                        // Indicar que no hay más trabajo
                        MPI_Send(nullptr, 0, MPI_CHAR, nodo_origen, TAG_SIN_TRABAJO, MPI_COMM_WORLD);
                    } else {
                        // Enviar mensaje para esperar
                        MPI_Send(nullptr, 0, MPI_CHAR, nodo_origen, TAG_SIN_TRABAJO, MPI_COMM_WORLD);
                    }
                    
                    break;
                }
                
                case TAG_RENDIMIENTO_NODO: {
                    // Recibir información de rendimiento
                    vector<char> buffer(tamanio_buffer);
                    MPI_Recv(buffer.data(), buffer.size(), MPI_CHAR, nodo_origen, TAG_RENDIMIENTO_NODO, MPI_COMM_WORLD, &estado);
                    
                    // Actualizar información de rendimiento
                    RendimientoNodo ren;
                    memcpy(&ren, buffer.data(), sizeof(RendimientoNodo));
                    
                    rendimiento_nodos[nodo_origen].tiempo_promedio_lote = ren.tiempo_promedio_lote;
                    rendimiento_nodos[nodo_origen].tiempo_comunicacion_mpi = ren.tiempo_comunicacion_mpi;
                    rendimiento_nodos[nodo_origen].carga_actual = ren.carga_actual;
                    rendimiento_nodos[nodo_origen].ultimo_reporte = chrono::system_clock::now();
                    
                    break;
                }
                
                case TAG_RESULTADO_FINAL: {
                    // Recibir notificación de finalización
                    MPI_Recv(nullptr, 0, MPI_CHAR, nodo_origen, TAG_RESULTADO_FINAL, MPI_COMM_WORLD, &estado);
                    
                    cout << "Nodo " << nodo_origen << " ha completado todo su procesamiento" << endl;
                    
                    // Comprobar si todos los nodos han terminado
                    esclavos_finalizados++;
                    cout << "→ Nodo " << nodo_origen << " ha finalizado." << endl;

                    if (esclavos_finalizados == num_nodos - 1) {
                        procesamiento_finalizado = true;
                        cout << "Todos los nodos esclavos han vinalizado. Preparando cierrre." << endl;
                    }

                    
                    break;
                }/*
                case TAG_LOTE_VOLUNTARIO: {
                    vector<char> buffer(tamanio_buffer);
                    MPI_Recv(buffer.data(), buffer.size(), MPI_CHAR, nodo_origen, TAG_LOTE_VOLUNTARIO, MPI_COMM_WORLD, &estado);

                    LoteTrabajo lote = deseriacllizarLote(buffer);
                    cola_trabajo.push(lote);

                    cout << "→ Maestro recibió lote voluntario desde nodo " << nodo_origen << endl;
                    break;
                }*/
               case TAG_NODO_DESOCUPADO: {
                    // Recibir la notificación (aunque no hay datos, se consume el mensaje)
                    MPI_Recv(nullptr, 0, MPI_CHAR, nodo_origen, TAG_NODO_DESOCUPADO, MPI_COMM_WORLD, &estado);
                    
                    // Solo contar si no había sido contado antes
                    if (nodos_desocupados.find(nodo_origen) == nodos_desocupados.end()) {
                        esclavos_desocupados++;
                    }
                    nodos_desocupados.insert(nodo_origen);
                    
                    cout << "→ Nodo " << nodo_origen << " notificado como desocupado (" 
                         << esclavos_desocupados << "/" << (num_nodos-1) << ")" << endl;
                    
                    // Si todos están desocupados, considerar finalización pronta
                    if (esclavos_desocupados >= (num_nodos - 1) && cola_trabajo.empty()) {
                        cout << "→ Todos los nodos están desocupados y no hay trabajo pendiente." << endl;
                        tiempo_ultima_actividad = ahora - chrono::seconds(TIMEOUT_FINALIZACION - 5); // Acelerar timeout
                    }
                    break;
                }

                
                default:
                    cout << "Mensaje desconocido recibido del nodo " << nodo_origen << " con tag " << tag << endl;
                    // Consumir el mensaje para evitar bloqueos
                    if (tamanio_buffer > 0) {
                        vector<char> buffer_dummy(tamanio_buffer);
                        MPI_Recv(buffer_dummy.data(), buffer_dummy.size(), MPI_CHAR, nodo_origen, tag, MPI_COMM_WORLD, &estado);
                    } else {
                        MPI_Recv(nullptr, 0, MPI_CHAR, nodo_origen, tag, MPI_COMM_WORLD, &estado);
                    }
                    break;
            }
        } else {


            // Dormir un poco para no saturar la CPU
            this_thread::sleep_for(chrono::milliseconds(10));
        }

        // Debug periódico del estado
        static auto ultimo_debug = chrono::system_clock::now();
        if (chrono::duration_cast<chrono::seconds>(ahora - ultimo_debug).count() >= 10) {
            cout << "→ [DEBUG] Estado actual: " 
                 << esclavos_finalizados << " finalizados, " 
                 << esclavos_desocupados << " desocupados, " 
                 << cola_trabajo.size() << " trabajos pendientes" << endl;
            ultimo_debug = ahora;
        }

    }
    
    cout << "→ Enviando señales de finalización a todos los nodos..." << endl;

    // Enviar señal de finalización a todos los nodos
    for (int i = 1; i < num_nodos; i++) {
        MPI_Send(nullptr, 0, MPI_CHAR, i, TAG_FINALIZAR, MPI_COMM_WORLD);
        cout << "→ Señal de finalización enviada al nodo " << i << endl;
    }
    
    // Imprimir resultados finales
    cout << "\n========== RESULTADOS FINALES ==========\n";
    chrono::duration<double> tiempo_total = chrono::system_clock::now() - tiempo_inicio;
    cout << "Tiempo total de procesamiento: " << tiempo_total.count() << " segundos" << endl;
    cout << "Total lotes procesados: " << lotes_completados << endl;
    
    imprimirEstadisticasWeb(estadisticas_globales, MAESTRO,web_sender);

    cout << "→ Maestro finalizando..." << endl;
    
}
