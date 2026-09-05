#include "VOTACION/common/config.hpp"
#include <omp.h>
#include <cstdlib>   // getenv, strtol
#include <cstring>   // strncpy

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

namespace config {

std::string leerEnv(const char* nombre, const std::string& por_defecto) {
    const char* valor = std::getenv(nombre);
    return (valor && *valor) ? std::string(valor) : por_defecto;
}

int leerEnvInt(const char* nombre, int por_defecto) {
    const char* valor = std::getenv(nombre);
    if (!valor || !*valor) return por_defecto;

    char* fin = nullptr;
    const long resultado = std::strtol(valor, &fin, 10);

    // Si la cadena no era un numero valido, nos quedamos con el valor por defecto
    // en lugar de arrancar con una configuracion silenciosamente incorrecta.
    if (fin == valor || *fin != '\0') {
        std::cerr << "[CONFIG] Aviso: " << nombre << "='" << valor
                  << "' no es un numero valido. Se usara " << por_defecto << ".\n";
        return por_defecto;
    }
    return static_cast<int>(resultado);
}

void imprimirConfiguracion() {
    std::cout << "\n--------- CONFIGURACION EFECTIVA ---------\n"
              << "  Datos de entrada    : " << ARCHIVO_ENTRADA_BASE << "<rank>.csv\n"
              << "  Dashboard           : " << URL_DASHBOARD << "\n"
              << "  Hilos OpenMP        : " << NUM_HILOS_POR_DEFECTO << "\n"
              << "  Hilos del algoritmo : " << NUM_HILOS_PARA_ALG << "\n"
              << "  Votos por lote      : " << TAM_LOTE_POR_DEFECTO << "\n"
              << "  Reporte cada        : " << INTERVALO_REPORTE_SEG << " s\n"
              << "  Balanceo cada       : " << INTERVALO_CHEQUEO_BALANCEO << " s\n"
              << "  Timeout de cierre   : " << TIMEOUT_FINALIZACION << " s\n"
              << "  Modo verboso        : " << (VERBOSE ? "SI" : "NO") << "\n"
              << "-----------------------------------------\n\n";
}

} // namespace config


CapacidadNodo detectarCapacidadNodo() {
    CapacidadNodo capacidad;
    capacidad.num_hilos               = omp_get_max_threads();
    capacidad.tiene_gpu               = false;
    capacidad.rendimiento_relativo    = 1.0f;
    capacidad.gpu_memoria_mb          = 0;
    capacidad.gpu_modelo[0]           = '\0';
    capacidad.velocidad_procesamiento = 0.0f;
    capacidad.lotes_pendientes        = 0;

#ifdef USE_CUDA
    int deviceCount = 0;
    const cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error == cudaSuccess && deviceCount > 0) {
        capacidad.tiene_gpu = true;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        capacidad.gpu_memoria_mb = static_cast<int>(deviceProp.totalGlobalMem / (1024 * 1024));
        std::strncpy(capacidad.gpu_modelo, deviceProp.name, sizeof(capacidad.gpu_modelo) - 1);
        capacidad.gpu_modelo[sizeof(capacidad.gpu_modelo) - 1] = '\0';

        // Heuristica: una GPU rinde al menos como 2 CPUs, mas un extra
        // proporcional al numero de multiprocesadores que tenga.
        capacidad.rendimiento_relativo = 2.0f + (deviceProp.multiProcessorCount / 20.0f);
    } else if (VERBOSE) {
        std::cerr << "[CUDA] No se detecto GPU utilizable: "
                  << cudaGetErrorString(error) << std::endl;
    }
#endif

    return capacidad;
}
