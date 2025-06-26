#include "VOTACION/common/config.hpp"
#include <omp.h>
#include <cstring> // para strncpy

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

CapacidadNodo detectarCapacidadNodo() {
    CapacidadNodo capacidad;
    capacidad.num_hilos = omp_get_max_threads();
    capacidad.tiene_gpu = false;
    capacidad.rendimiento_relativo = 1.0;
    capacidad.gpu_memoria_mb = 0;
    capacidad.gpu_modelo[0] = '\0'; // Inicializar string vacío
    capacidad.velocidad_procesamiento = 0.0;
    capacidad.lotes_pendientes = 0;

#ifdef USE_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    std::cerr << "[CUDA DEBUG] cudaGetDeviceCount returned: " << cudaGetErrorString(error) << std::endl;

    if (error == cudaSuccess && deviceCount > 0) {
        capacidad.tiene_gpu = true;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        capacidad.gpu_memoria_mb = deviceProp.totalGlobalMem / (1024 * 1024);
        strncpy(capacidad.gpu_modelo, deviceProp.name, sizeof(capacidad.gpu_modelo));
        capacidad.gpu_modelo[sizeof(capacidad.gpu_modelo) - 1] = '\0'; // evitar overflow
        capacidad.rendimiento_relativo = 2.0 + (deviceProp.multiProcessorCount / 20.0);
    }
#endif


    return capacidad;
}
