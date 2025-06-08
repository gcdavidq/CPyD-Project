#include "balanceo_carga.hpp"
#include <mpi.h>
#include <chrono>
#include <iostream>

using namespace std;
void balanceoCarga(std::vector<RendimientoNodo>& rendimiento_nodos, std:: queue<LoteTrabajo>& cola_trabajo, int num_nodos){

    // Obtener el tiempo actual
    auto ahora = chrono::system_clock::now();
    EstadoBalanceo estado_balanceo;
    estado_balanceo.rendimiento_nodos = rendimiento_nodos;
    estado_balanceo.timestamp = ahora;
    estado_balanceo.lotes_balanceados = 0;

    vector<int> nodos_sobrecargados;
    vector<int> nodos_baja_carga;

    // Identificar nodos sobrecargados y con baja carga
    for (int i =1; i<num_nodos; i++){
        // Nodo sobrecargado si su carga actual es mayot al 80%
        if (rendimiento_nodos[i].carga_actual > 80.0f){
            nodos_sobrecargados.push_back(i);
        } else if (rendimiento_nodos[i].carga_actual<30.0f && 
                   rendimiento_nodos[i].lotes_asignados < cola_trabajo.size() + 5){
                    nodos_baja_carga.push_back(i);

                   }
        
    }

    //Reasignar trabajo si hay nodos sobrecargados y con baja carga
    if (!nodos_sobrecargados.empty() && !nodos_baja_carga.empty()){
        for (int nodo_origen : nodos_sobrecargados){

            //Solicitar transferencia de trabajo
            MPI_Send(nullptr, 0, MPI_CHAR, nodo_origen, TAG_BALANCE_CARGA, MPI_COMM_WORLD);

            MPI_Status estado;
            int tamanio_buffer;
            MPI_Probe(nodo_origen, TAG_ENVIO_TRABAJO, MPI_COMM_WORLD, &estado);

            MPI_Get_count(&estado, MPI_CHAR, &tamanio_buffer);

            vector<char> buffer(tamanio_buffer);
            MPI_Recv(buffer.data(), buffer.size(), MPI_CHAR, nodo_origen, TAG_ENVIO_TRABAJO, MPI_COMM_WORLD, &estado);

            LoteTrabajo lote = deserealizarLote(buffer);

            //Enviar a un nodo con baja carga
            int nodo_destino = nodos_baja_carga[0];
            
            vector<char> buffer_envio;
            serializarLote(lote, buffer_envio);

            MPI_Send(buffer_envio.data(), buffer_envio.size(), MPI_CHAR, nodo_destino, TAG_ENVIO_TRABAJO, MPI_COMM_WORLD);

            //Actualizar estadisticas de balanceo
            rendimiento_nodos[nodo_origen].lotes_asignados--;
            rendimiento_nodos[nodo_destino].lotes_asignados++;

            estado_balanceo.reasignaciones[nodo_origen]++;
            estado_balanceo.lotes_balanceados++;

            //rotar nodo destino 
            nodos_baja_carga.push_back(nodo_destino);
            nodos_baja_carga.erase(nodos_baja_carga.begin());


            
        }
        // Imprimir estadisticas de balanceo
        estado_balanceo.imprimirGraficoBalanceo();
    }

    }