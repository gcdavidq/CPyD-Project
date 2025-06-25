#include "VOTACION/balanceo/balanceo_carga.hpp"

/*
SCRIPT para ejecutar el balanceo de carga - VERSIÓN CORREGIDA
*/
#include "balanceo_carga.hpp"
//#include <mpi.h>
#include <chrono>
#include <iostream>
#include <set>
#include <algorithm>

using namespace std;

void balanceoCarga(std::vector<RendimientoNodo>& rendimiento_nodos, 
                   std::queue<LoteTrabajo>& cola_trabajo, 
                   int num_nodos, 
                   const set<int>& nodos_desocupados) {

    // Obtener el tiempo actual
    auto ahora = chrono::system_clock::now();
    EstadoBalanceo estado_balanceo;
    estado_balanceo.rendimiento_nodos = rendimiento_nodos;
    estado_balanceo.timestamp = ahora;
    estado_balanceo.lotes_balanceados = 0;

    vector<int> nodos_sobrecargados;
    vector<int> nodos_baja_carga;

    // Identificar nodos sobrecargados y con baja carga
    for (int i = 1; i < num_nodos; i++) {
        if (rendimiento_nodos[i].carga_actual > 80.0f) {
            nodos_sobrecargados.push_back(i);
            cout << "[DEBUG] Nodo " << i << " sobrecargado: " << rendimiento_nodos[i].carga_actual << "%" << endl;
        } else if (
            rendimiento_nodos[i].carga_actual < 30.0f &&
            rendimiento_nodos[i].lotes_asignados < cola_trabajo.size() + 5
        ) {
            nodos_baja_carga.push_back(i);
            cout << "[DEBUG] Nodo " << i << " con baja carga: " << rendimiento_nodos[i].carga_actual << "%" << endl;
        } else if (nodos_desocupados.count(i)) {
            nodos_baja_carga.push_back(i);
            cout << "[DEBUG] Nodo " << i << " desocupado agregado a baja carga" << endl;
        }
    }

    cout << "[INFO] Nodos sobrecargados encontrados: " << nodos_sobrecargados.size() << endl;
    cout << "[INFO] Nodos con baja carga encontrados: " << nodos_baja_carga.size() << endl;

    // CORRECCIÓN
    if (nodos_sobrecargados.empty() || nodos_baja_carga.empty()) {
        std::cout << "[INFO] No hay condiciones para realizar balanceo de carga.\n";
        cout << "  - Nodos sobrecargados: " << nodos_sobrecargados.size() << endl;
        cout << "  - Nodos con baja carga: " << nodos_baja_carga.size() << endl;
        return; // Salir temprano
    }

    // Aquí SÍ hay condiciones para balancear
    std::cout << "[INFO] ¡Iniciando balanceo de carga!\n";
    cout << "  - Balanceando entre " << nodos_sobrecargados.size() 
         << " nodos sobrecargados y " << nodos_baja_carga.size() 
         << " nodos con baja carga" << endl;

    // Crear una cola circular para rotar nodos destino de manera segura
    int indice_destino = 0;
    
    // Procesar cada nodo sobrecargado
    for (int nodo_origen : nodos_sobrecargados) {
        cout << "[BALANCEO] Procesando nodo sobrecargado: " << nodo_origen << endl;
        
        // Limitar transferencias por nodo para evitar vaciarlo completamente
        int max_transferencias = max(1, rendimiento_nodos[nodo_origen].lotes_asignados / 4);
        int transferencias_realizadas = 0;

        while (transferencias_realizadas < max_transferencias && 
               rendimiento_nodos[nodo_origen].lotes_asignados > 0 &&
               !nodos_baja_carga.empty()) {
            
            try {
                // CORRECCIÓN: Simulación segura sin MPI real
                cout << "[SIMULACION] Solicitando transferencia de trabajo al nodo " << nodo_origen << endl;
                
                // En lugar de usar MPI, simulamos la transferencia
                // Crear un lote simulado para transferir
                LoteTrabajo lote_simulado;
                lote_simulado.id_lote = transferencias_realizadas + (nodo_origen * 1000);
                lote_simulado.inicio_timestamp = "2024-12-16T09:00:00";
                lote_simulado.fin_timestamp = "2024-12-16T09:01:00";
                
                // Seleccionar nodo destino de manera segura
                int nodo_destino = nodos_baja_carga[indice_destino % nodos_baja_carga.size()];
                
                // Verificar que el nodo destino no se sobrecargue
                if (rendimiento_nodos[nodo_destino].carga_actual + 10.0f > 80.0f) {
                    cout << "[WARNING] Nodo destino " << nodo_destino 
                         << " se sobrecargaría. Buscando alternativa..." << endl;
                    
                    // Buscar otro nodo destino
                    bool encontrado = false;
                    for (size_t j = 0; j < nodos_baja_carga.size(); j++) {
                        int nodo_alt = nodos_baja_carga[j];
                        if (rendimiento_nodos[nodo_alt].carga_actual + 10.0f <= 80.0f) {
                            nodo_destino = nodo_alt;
                            encontrado = true;
                            break;
                        }
                    }
                    
                    if (!encontrado) {
                        cout << "[WARNING] No hay nodos destino disponibles sin sobrecargar" << endl;
                        break;
                    }
                }

                cout << "[BALANCEO] Transfiriendo lote " << lote_simulado.id_lote 
                     << " del nodo " << nodo_origen << " al nodo " << nodo_destino << endl;

                // Actualizar estadísticas de balanceo
                rendimiento_nodos[nodo_origen].lotes_asignados--;
                rendimiento_nodos[nodo_destino].lotes_asignados++;
                
                // Actualizar carga estimada
                rendimiento_nodos[nodo_origen].carga_actual -= 5.0f;
                rendimiento_nodos[nodo_destino].carga_actual += 5.0f;

                estado_balanceo.reasignaciones[nodo_origen]++;
                estado_balanceo.lotes_balanceados++;

                // CORRECCIÓN: Rotación segura del índice destino
                indice_destino = (indice_destino + 1) % nodos_baja_carga.size();
                
                transferencias_realizadas++;
                
                cout << "[SUCCESS] Transferencia " << transferencias_realizadas 
                     << " completada exitosamente" << endl;

            } catch (const std::exception& e) {
                cout << "[ERROR] Error durante transferencia: " << e.what() << endl;
                break;
            } catch (...) {
                cout << "[ERROR] Error desconocido durante transferencia" << endl;
                break;
            }
        }
        
        cout << "[INFO] Nodo " << nodo_origen << " procesado. Transferencias realizadas: " 
             << transferencias_realizadas << endl;
    }

    // Actualizar el estado de balanceo con los nuevos valores
    estado_balanceo.rendimiento_nodos = rendimiento_nodos;
    
    // Imprimir estadísticas de balanceo
    cout << "[INFO] Balanceo completado. Imprimiendo estadísticas..." << endl;
    estado_balanceo.imprimirGraficoBalanceo();
    
    cout << "[RESUMEN] Total de lotes balanceados: " << estado_balanceo.lotes_balanceados << endl;
}