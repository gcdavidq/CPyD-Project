/*
Ejecutable principal del proyecto. 
Se realizar una llamada a la función de ejecución del nodo maestro o esclavo según el identificador del proceso.
*/

#include <iostream>
#include <cstdlib>
#include <omp.h>
#include <mpi.h>
#include <VOTACION/common/config.hpp> 
#include <VOTACION/maestro/nodo_maestro.hpp>
#include <VOTACION/esclavo/nodo_esclavo.hpp>

int main(int argc, char** argv) {
    // Inicializar entorno MPI con soporte para hilos múltiples
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE) {
        std::cerr << "Advertencia: El entorno MPI no proporciona soporte completo para hilos." << std::endl;
    }

    // Obtener el identificador de proceso y número total de procesos
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Configurar OpenMP
    int num_hilos = NUM_HILOS_POR_DEFECTO;
    if (argc > 1) {
        num_hilos = std::atoi(argv[1]);
    }
    omp_set_num_threads(num_hilos);

    // Ejecutar función correspondiente según el tipo de nodo
    if (rank == MAESTRO) {
        ejecutarNodoMaestro(size);
    } else {
        ejecutarNodoEsclavo(rank, MAESTRO);
    }

    // Finalizar entorno MPI
    MPI_Finalize();
    return 0;
}
