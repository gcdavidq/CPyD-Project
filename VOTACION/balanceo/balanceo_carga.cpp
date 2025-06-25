/*
SCRIPT para ejecutar el balanceo de carga
*/
#include "balanceo_carga.hpp"
#include <mpi.h>
#include <chrono>
#include <iostream>
#include <set>
#include <algorithm>

using namespace std;
void balanceoCarga(std::vector<RendimientoNodo>& rendimiento_nodos, 
                   std::queue<LoteTrabajo>& cola_trabajo, 
                   int num_nodos, 
                   const set<int>& nodos_desocupados) {

    auto ahora = chrono::system_clock::now();
    EstadoBalanceo estado_balanceo;
    estado_balanceo.rendimiento_nodos = rendimiento_nodos;
    estado_balanceo.timestamp = ahora;
    estado_balanceo.lotes_balanceados = 0;

    vector<int> nodos_sobrecargados;
    vector<int> nodos_baja_carga;
    // Primero: Agregar nodos desocupados a nodos_baja_carga
    for (int i = 1; i < num_nodos; i++) {
        if (nodos_desocupados.count(i)) {
            nodos_baja_carga.push_back(i);
        }
    }

    // Segundo: Evaluar el resto de nodos según su carga actual
    for (int i = 1; i < num_nodos; i++) {
        if (nodos_desocupados.count(i)) continue; // ya lo agregamos arriba

        float carga = rendimiento_nodos[i].carga_actual;

        if (carga > 50.0f) {
            nodos_sobrecargados.push_back(i);
        } else if (carga < 30.0f) {
            nodos_baja_carga.push_back(i);
        }
    }

    

    // CUANDO NO HAY CONDICIONES PARA HACER BALANCEO
    if (nodos_sobrecargados.empty() || nodos_baja_carga.empty()) {
        std::cout << "[INFO] No hay condiciones para realizar balanceo de carga.\n";
        std::cout << "  - Nodos sobrecargados: " << nodos_sobrecargados.size() << std::endl;
        std::cout << "  - Nodos con baja carga: " << nodos_baja_carga.size() << std::endl;
        return; 
    }

    // CUANDO SE NECESITA HACER BALANCEO
    std::cout << "[INFO] Iniciando balanceo de carga entre " 
              << nodos_sobrecargados.size() << " nodos sobrecargados y " 
              << nodos_baja_carga.size() << " nodos con baja carga.\n";

    std::cout <<"NODOS CON SOBRECARGA";
    std::cout << " (IDs: ";
        for (int id : nodos_sobrecargados) std::cout << id << " ";
        std::cout << ")";

    std::cout<<"NODOS CON BAJA CARGA";
    std::cout << " (IDs: ";
        for (int id : nodos_baja_carga) std::cout << id << " ";
        std::cout << ")";
 
    for (int nodo_origen : nodos_sobrecargados) {
        // Verificación de seguridad
        /*
        if (rendimiento_nodos[nodo_origen].lotes_asignados <= 0) {
            continue; // Saltar si no hay lotes para transferir
        } */

        try {
           
            MPI_Send(nullptr, 0, MPI_CHAR, nodo_origen, TAG_BALANCE_CARGA, MPI_COMM_WORLD);

            MPI_Status estado;
            int tamanio_buffer;
            MPI_Probe(nodo_origen, TAG_ENVIO_TRABAJO, MPI_COMM_WORLD, &estado);
            MPI_Get_count(&estado, MPI_CHAR, &tamanio_buffer);
            

            vector<char> buffer(tamanio_buffer);
            MPI_Recv(buffer.data(), buffer.size(), MPI_CHAR, nodo_origen, 
                    TAG_ENVIO_TRABAJO, MPI_COMM_WORLD, &estado);
            

            LoteTrabajo lote = deserializarLote(buffer);
            cout<<"LLEGARON VOTOS A REDISTRIBUIR DEL NODO: "<< nodo_origen
                << "CON UNA CANTIDAD DE: "<< lote.votos.size() <<endl;
            


            // Enviar a un nodo con baja carga (con verificación)
            if (nodos_baja_carga.empty()) {
                std::cout << "[WARNING] No hay nodos de baja carga disponibles\n";
                break;
            }

            int nodo_destino = nodos_baja_carga[0];
            cout<<"NODO SELECCIONADO PARA BALANCEO DE CARGA: " << nodo_destino;
            
            vector<char> buffer_envio;
            serializarLote(lote, buffer_envio);

            MPI_Send(buffer_envio.data(), buffer_envio.size(), MPI_CHAR, 
                    nodo_destino, TAG_ENVIO_TRABAJO, MPI_COMM_WORLD);

            // Actualizar estadísticas
            rendimiento_nodos[nodo_origen].lotes_asignados--;
            rendimiento_nodos[nodo_destino].lotes_asignados++;

            estado_balanceo.reasignaciones[nodo_origen]++;
            estado_balanceo.lotes_balanceados++;

            // ✅ Rotación más segura
            if (nodos_baja_carga.size() > 1) {
                rotate(nodos_baja_carga.begin(), 
                       nodos_baja_carga.begin() + 1, 
                       nodos_baja_carga.end());
            }

        } catch (const std::exception& e) {
            std::cout << "[ERROR] Error en balanceo MPI: " << e.what() << std::endl;
            break;
        }
    }

    // Imprimir estadísticas
    estado_balanceo.imprimirGraficoBalanceo();
}
