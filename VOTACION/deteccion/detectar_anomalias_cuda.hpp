/**
 * @file detectar_anomalias_cuda.hpp
 * @brief Header para detección de anomalías usando CUDA GPU
 * @author Sistema de Detección de Fraudes Electorales
 * @date 2025
 * 
 * Este archivo contiene las declaraciones para la implementación CUDA
 * de detección de anomalías en votos electorales. Mantiene compatibilidad
 * con la estructura de datos original pero optimizada para GPU.
 */

#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "VOTACION/common/estructura_votos.hpp"

namespace deteccion_cuda {

/**
 * @brief Configuración para el procesamiento en GPU
 */
struct ConfiguracionCUDA {
    int ventana_datos_por_hilo;     // Cantidad de datos por hilo (25, 50, 100)
    int threads_per_block;          // Threads por bloque CUDA
    int max_blocks;                 // Máximo número de bloques
    bool usar_memoria_compartida;   // Optimización con shared memory
    bool debug_mode;                // Modo debug para información adicional
    
    // Constructor con valores por defecto
    ConfiguracionCUDA() : 
        ventana_datos_por_hilo(50),
        threads_per_block(256),
        max_blocks(65535),
        usar_memoria_compartida(true),
        debug_mode(false) {}
};

/**
 * @brief Estructura de voto optimizada para GPU
 * Mantiene compatibilidad con la estructura original pero optimizada para CUDA
 */
struct VotoCUDA {
    char timestamp[20];      // Timestamp como array de char para GPU
    char region[10];         // Región como array de char
    char dni[12];           // DNI como array de char
    char candidato[10];     // Candidato como array de char
    bool anomalo;           // Si es anómalo real
    bool anomalia_detectada; // Si fue detectado como anómalo
    int tipo_anomalia;      // Tipo de anomalía detectada (-1: ninguna, 1: DNI, 2: concentración, 3: flujo)
};

/**
 * @brief Contadores globales para estadísticas en GPU
 */
struct ContadoresGlobales {
    int* contador_dni;              // Hash map simplificado para DNIs
    int* contador_region_candidato; // Hash map para region|candidato
    int* contador_timestamp;        // Hash map para timestamps por minuto
    int hash_size_dni;             // Tamaño del hash para DNIs
    int hash_size_region_cand;     // Tamaño del hash para región-candidato
    int hash_size_timestamp;       // Tamaño del hash para timestamps
};

/**
 * @brief Resultado de detección compatible con la versión CPU
 */

struct ResultadoDeteccionCUDA {
    std::vector<Voto> validos;              // Votos válidos
    std::vector<Voto> anomalos;             // Votos anómalos
    double tiempo_proceso_ms;               // Tiempo total de procesamiento
    int anomalias_flujo_excesivo;          // Contador de anomalías de flujo
    int anomalias_concentracion;           // Contador de anomalías de concentración
    int anomalias_duplicados;              // Contador de anomalías de DNI duplicado
    double precision;                       // Precisión del modelo
    double recall;                         // Recall del modelo
    double f1_score;                       // F1-Score del modelo
    
    // Información adicional de GPU
    int bloques_utilizados;                // Número de bloques CUDA utilizados
    int hilos_por_bloque;                 // Hilos por bloque utilizados
    int datos_por_hilo;                   // Ventana de datos por hilo
    float memoria_gpu_utilizada_mb;       // Memoria GPU utilizada en MB
};

/**
 * @brief Información del dispositivo GPU
 */
struct InfoGPU {
    int device_id;
    char nombre[256];
    size_t memoria_total;
    size_t memoria_libre;
    int multiprocessors;
    int max_threads_per_block;
    int max_blocks_per_grid;
    int compute_capability_major;
    int compute_capability_minor;
};

/**
 * @brief Detecta anomalías usando CUDA GPU con ventanas de datos configurables
 * 
 * @param votos Vector de votos a analizar
 * @param config Configuración para el procesamiento CUDA
 * @return ResultadoDeteccionCUDA con los resultados y métricas
 */
ResultadoDeteccionCUDA detectarAnomaliasCUDA(
    const std::vector<Voto>& votos, 
    const ConfiguracionCUDA& config = ConfiguracionCUDA()
);

/**
 * @brief Obtiene información del dispositivo GPU disponible
 * 
 * @param device_id ID del dispositivo (por defecto 0)
 * @return InfoGPU con información del dispositivo
 */
InfoGPU obtenerInfoGPU(int device_id = 0);

/**
 * @brief Calcula la configuración óptima de bloques y threads
 * 
 * @param total_votos Número total de votos
 * @param datos_por_hilo Ventana de datos por hilo
 * @return ConfiguracionCUDA optimizada
 */
ConfiguracionCUDA calcularConfiguracionOptima(int total_votos, int datos_por_hilo = 50);

/**
 * @brief Libera memoria GPU utilizada
 */
void limpiarMemoriaGPU();

// Funciones utilitarias para manejo de errores CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Error CUDA en %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Error en kernel CUDA en %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
        cudaDeviceSynchronize(); \
    } while(0)

} // namespace deteccion_cuda
