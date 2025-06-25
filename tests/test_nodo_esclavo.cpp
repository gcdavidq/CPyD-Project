#include <mpi.h>
#include <iostream>
#include <vector>
#include "VOTACION/esclavo/nodo_esclavo.hpp"
#include "VOTACION/protocolo/protocolo.hpp"
#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/common/config.hpp"

using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int maestro = 0;
    const int esclavo = 1;

    if (size < 2) {
        if (rank == 0) {
            cerr << "Este test requiere al menos 2 procesos MPI (maestro + esclavo)." << endl;
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == maestro) {
        // MAESTRO: espera mensajes del esclavo
        MPI_Status status;

        // 1. Recibir capacidad del nodo esclavo
        vector<char> buffer_cap(sizeof(CapacidadNodo));
        MPI_Recv(buffer_cap.data(), buffer_cap.size(), MPI_CHAR, esclavo, TAG_CAPACIDAD_NODO, MPI_COMM_WORLD, &status);
        cout << "[Maestro] Capacidad del nodo esclavo recibida correctamente." << endl;

        // 2. Recibir al menos una estadística de procesamiento
        vector<char> buffer_stats(1024); // tamaño arbitrario seguro
        MPI_Recv(buffer_stats.data(), buffer_stats.size(), MPI_CHAR, esclavo, TAG_REPORTE_STATS, MPI_COMM_WORLD, &status);
        Estadisticas stats = deserializarEstadisticas(buffer_stats);
        cout << "[Maestro] Estadísticas recibidas:" << endl;
        cout << " - Votos: " << stats.total_votos << endl;
        cout << " - Anomalías detectadas: " << stats.anomalias_detectadas << endl;

        // 3. Esperar señal de finalización
        MPI_Recv(nullptr, 0, MPI_CHAR, esclavo, TAG_RESULTADO_FINAL, MPI_COMM_WORLD, &status);
        cout << "[Maestro] Nodo esclavo finalizó correctamente." << endl;
    }

    else if (rank == esclavo) {
        // ESCLAVO: usar una ruta CSV fija de test
        string ruta = "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/DATA/votos_simulados_region1.csv";
        ejecutarNodoEsclavo(esclavo, maestro);  // función ya implementada y modularizada
    }

    MPI_Finalize();
    return 0;
}
