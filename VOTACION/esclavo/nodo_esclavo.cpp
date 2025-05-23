/*
Script que incluye todas las librerías necesarias para la simulación de un nodo esclavo en un sistema de votación.

POR AHORA, solo contiene la simulación de la llegada de votos y la detección de anomalías.
El nodo esclavo recibe un CSV con los votos y un ID de nodo, simula la llegada de los votos en lotes y detecta anomalías en ellos.
*/
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstring>
#include <iomanip>
#include <ctime>
#include <random>
#include <mpi.h>
#include "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/VOTACION/simulacion/simulacion_llegada.hpp"
#include "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/VOTACION/deteccion/detectar_anomalias.hpp"

using namespace std;

using deteccion::ResultadoDeteccion; //Usamos el namespace deteccion para acceder a la estructura ResultadoDeteccion

int main(int argc, char* argv[])
{
    if (argc < 3) { //en caso de que no se pasen los argumentos necesarios
        cerr << "Uso: nodo_esclavo <csv> <nodo_id>\n";
        return 1;
    }
    string ruta_csv = argv[1]; //argv [1] es el CSV
    int nodo_id = std::stoi(argv[2]);

    // 1. Cargar votos
    auto todos = leerVotos(ruta_csv);

    // 2. Cola thread-safe para comunicar productor → consumidor
    queue<vector<Voto>> cola;
    mutex m;
    bool done = false;

    // 3. Hilo productor: simula llegada
    thread productor([&]{
        simularLlegadaVotos(todos, nodo_id,
            [&](vector<Voto>&& lote){
                lock_guard<mutex> lk(m);
                cola.push(move(lote));
            });
        done = true;
    });

    // 4. Hilo consumidor: detecta anomalías
    thread consumidor([&]{
        while (!done || !cola.empty()) {
            vector<Voto> lote;
            {
                lock_guard<mutex> lk(m);
                if (cola.empty()) {
                    this_thread::sleep_for(chrono::milliseconds(30));
                    continue;
                }
                lote = move(cola.front()); cola.pop();
            }
            ResultadoDeteccion R = deteccion::detectarAnomaliasCPU(lote); // CPU o CUDA
            cout << "[Nodo "<< nodo_id << "] "
                 << "validos="  << R.validos.size()
                 << " anom="    << R.anomalos.size()
                 << " t="       << R.tiempo_proceso_ms << " ms\n";

            //   aquí se enviará R al maestro con MPI_Send
        }
    });

    productor.join();
    consumidor.join();
    return 0;
}
