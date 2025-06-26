/**
 * @file detectar_anomalias_cuda.cu
 * @brief Implementación CUDA para detección de anomalías en votos electorales
 * @author Sistema de Detección de Fraudes Electorales
 * @date 2025
 * 
 * Esta implementación utiliza CUDA para procesar grandes volúmenes de datos
 * de votos electorales en paralelo, detectando tres tipos de anomalías:
 * 1. Flujo excesivo de datos (demasiados votos en poco tiempo)
 * 2. Concentración de votos a un candidato (dominio excesivo)
 * 3. Repetición de DNI (mismo DNI vota múltiples veces)
 */


#include "VOTACION/deteccion/detectar_anomalias_cuda.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <cmath>
#include <numeric>

namespace deteccion_cuda {

/**
 * @brief Función hash simple para strings en GPU
 * @param str String a hashear
 * @param len Longitud del string
 * @return Valor hash
 */
__device__ unsigned int hash_string_gpu(const char* str, int len) {
    unsigned int hash = 5381;
    for (int i = 0; i < len && str[i] != '\0'; i++) {
        hash = ((hash << 5) + hash) + str[i];
    }
    return hash;
}

/**
 * @brief Convierte estructura Voto a VotoCUDA
 * @param voto_original Voto original
 * @param voto_cuda Voto CUDA de salida
 */
__host__ void convertirVotoACUDA(const Voto& voto_original, VotoCUDA& voto_cuda) {
    // Copiar strings limitando la longitud
    strncpy(voto_cuda.timestamp, voto_original.timestamp.c_str(), 19);
    voto_cuda.timestamp[19] = '\0';
    
    strncpy(voto_cuda.region, voto_original.region.c_str(), 9);
    voto_cuda.region[9] = '\0';
    
    strncpy(voto_cuda.dni, voto_original.dni.c_str(), 11);
    voto_cuda.dni[11] = '\0';
    
    strncpy(voto_cuda.candidato, voto_original.candidato.c_str(), 9);
    voto_cuda.candidato[9] = '\0';
    
    voto_cuda.anomalo = voto_original.anomalo;
    voto_cuda.anomalia_detectada = false;
    voto_cuda.tipo_anomalia = -1;
}

/**
 * @brief Convierte VotoCUDA de vuelta a Voto
 * @param voto_cuda Voto CUDA
 * @param voto_original Voto original de salida
 */
__host__ void convertirVotoDeCUDA(const VotoCUDA& voto_cuda, Voto& voto_original) {
    voto_original.timestamp = std::string(voto_cuda.timestamp);
    voto_original.region = std::string(voto_cuda.region);
    voto_original.dni = std::string(voto_cuda.dni);
    voto_original.candidato = std::string(voto_cuda.candidato);
    voto_original.anomalo = voto_cuda.anomalo;
    voto_original.anomalia_detectada = voto_cuda.anomalia_detectada;
    voto_original.tipo_anomalia = voto_cuda.tipo_anomalia;
}

/**
 * @brief Kernel para construir contadores globales (Fase 1)
 * @param votos Array de votos en GPU
 * @param total_votos Número total de votos
 * @param contador_dni Contador global de DNIs
 * @param contador_region_candidato Contador global región-candidato
 * @param contador_timestamp Contador global de timestamps
 * @param hash_size_dni Tamaño del hash DNI
 * @param hash_size_region_cand Tamaño del hash región-candidato
 * @param hash_size_timestamp Tamaño del hash timestamp
 * @param datos_por_hilo Ventana de datos por hilo
 */
__global__ void construirContadoresGlobales(
    VotoCUDA* votos,
    int total_votos,
    int* contador_dni,
    int* contador_region_candidato,
    int* contador_timestamp,
    int hash_size_dni,
    int hash_size_region_cand,
    int hash_size_timestamp,
    int datos_por_hilo
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int inicio = tid * datos_por_hilo;
    int fin = min(inicio + datos_por_hilo, total_votos);
    
    // Cada hilo procesa su ventana de datos
    for (int i = inicio; i < fin; i++) {
        VotoCUDA& voto = votos[i];
        
        // Contar DNI
        unsigned int hash_dni = hash_string_gpu(voto.dni, 11) % hash_size_dni;
        atomicAdd(&contador_dni[hash_dni], 1);
        
        // Contar región-candidato (concatenar strings)
        char region_candidato[20];
        int len_region = 0;
        while (voto.region[len_region] != '\0' && len_region < 9) len_region++;
        int len_candidato = 0;
        while (voto.candidato[len_candidato] != '\0' && len_candidato < 9) len_candidato++;
        
        for (int j = 0; j < len_region; j++) {
            region_candidato[j] = voto.region[j];
        }
        region_candidato[len_region] = '|';
        for (int j = 0; j < len_candidato; j++) {
            region_candidato[len_region + 1 + j] = voto.candidato[j];
        }
        region_candidato[len_region + 1 + len_candidato] = '\0';
        
        unsigned int hash_region_cand = hash_string_gpu(region_candidato, len_region + 1 + len_candidato) % hash_size_region_cand;
        atomicAdd(&contador_region_candidato[hash_region_cand], 1);
        
        // Contar timestamp (por minuto - primeros 16 caracteres)
        char timestamp_minuto[17];
        for (int j = 0; j < 16; j++) {
            timestamp_minuto[j] = voto.timestamp[j];
        }
        timestamp_minuto[16] = '\0';
        
        unsigned int hash_timestamp = hash_string_gpu(timestamp_minuto, 16) % hash_size_timestamp;
        atomicAdd(&contador_timestamp[hash_timestamp], 1);
    }
}

/**
 * @brief Kernel para detectar anomalías (Fase 2)
 * @param votos Array de votos en GPU
 * @param total_votos Número total de votos
 * @param contador_dni Contador global de DNIs
 * @param contador_region_candidato Contador global región-candidato
 * @param contador_timestamp Contador global de timestamps
 * @param hash_size_dni Tamaño del hash DNI
 * @param hash_size_region_cand Tamaño del hash región-candidato
 * @param hash_size_timestamp Tamaño del hash timestamp
 * @param umbral_concentracion Umbral para concentración
 * @param umbral_flujo Umbral para flujo excesivo
 * @param datos_por_hilo Ventana de datos por hilo
 * @param contador_duplicados Contador de duplicados (salida)
 * @param contador_concentracion Contador de concentración (salida)
 * @param contador_flujo Contador de flujo (salida)
 */
__global__ void detectarAnomalias(
    VotoCUDA* votos,
    int total_votos,
    int* contador_dni,
    int* contador_region_candidato,
    int* contador_timestamp,
    int hash_size_dni,
    int hash_size_region_cand,
    int hash_size_timestamp,
    int umbral_concentracion,
    int umbral_flujo,
    int datos_por_hilo,
    int* contador_duplicados,
    int* contador_concentracion,
    int* contador_flujo
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int inicio = tid * datos_por_hilo;
    int fin = min(inicio + datos_por_hilo, total_votos);
    
    int local_duplicados = 0;
    int local_concentracion = 0;
    int local_flujo = 0;
    
    // Cada hilo procesa su ventana de datos
    for (int i = inicio; i < fin; i++) {
        VotoCUDA& voto = votos[i];
        bool es_anomalo = false;
        int tipo = -1;
        
        // Verificar DNI duplicado
        unsigned int hash_dni = hash_string_gpu(voto.dni, 11) % hash_size_dni;
        if (contador_dni[hash_dni] > 1) {
            es_anomalo = true;
            tipo = 1;
            local_duplicados++;
        }
        
        // Verificar concentración
        char region_candidato[20];
        int len_region = 0;
        while (voto.region[len_region] != '\0' && len_region < 9) len_region++;
        int len_candidato = 0;
        while (voto.candidato[len_candidato] != '\0' && len_candidato < 9) len_candidato++;
        
        for (int j = 0; j < len_region; j++) {
            region_candidato[j] = voto.region[j];
        }
        region_candidato[len_region] = '|';
        for (int j = 0; j < len_candidato; j++) {
            region_candidato[len_region + 1 + j] = voto.candidato[j];
        }
        region_candidato[len_region + 1 + len_candidato] = '\0';
        
        unsigned int hash_region_cand = hash_string_gpu(region_candidato, len_region + 1 + len_candidato) % hash_size_region_cand;
        if (contador_region_candidato[hash_region_cand] > umbral_concentracion) {
            if (!es_anomalo) {
                tipo = 2;
            }
            es_anomalo = true;
            local_concentracion++;
        }
        
        // Verificar flujo excesivo
        char timestamp_minuto[17];
        for (int j = 0; j < 16; j++) {
            timestamp_minuto[j] = voto.timestamp[j];
        }
        timestamp_minuto[16] = '\0';
        
        unsigned int hash_timestamp = hash_string_gpu(timestamp_minuto, 16) % hash_size_timestamp;
        if (contador_timestamp[hash_timestamp] > umbral_flujo) {
            if (!es_anomalo) {
                tipo = 3;
            }
            es_anomalo = true;
            local_flujo++;
        }
        
        voto.anomalia_detectada = es_anomalo;
        voto.tipo_anomalia = tipo;
    }
    
    // Reducir contadores locales a globales
    atomicAdd(contador_duplicados, local_duplicados);
    atomicAdd(contador_concentracion, local_concentracion);
    atomicAdd(contador_flujo, local_flujo);
}

InfoGPU obtenerInfoGPU(int device_id) {
    InfoGPU info;
    cudaDeviceProp prop;
    
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    info.device_id = device_id;
    strncpy(info.nombre, prop.name, 255);
    info.nombre[255] = '\0';
    info.memoria_total = prop.totalGlobalMem;
    info.multiprocessors = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_blocks_per_grid = prop.maxGridSize[0];
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    
    size_t libre, total;
    CUDA_CHECK(cudaMemGetInfo(&libre, &total));
    info.memoria_libre = libre;
    
    return info;
}

ConfiguracionCUDA calcularConfiguracionOptima(int total_votos, int datos_por_hilo) {
    ConfiguracionCUDA config;
    InfoGPU info = obtenerInfoGPU();
    
    config.ventana_datos_por_hilo = datos_por_hilo;
    
    // Calcular número de hilos necesarios
    int hilos_necesarios = (total_votos + datos_por_hilo - 1) / datos_por_hilo;
    
    // Configurar threads por bloque (múltiplo de 32 para eficiencia)
    config.threads_per_block = std::min(256, ((hilos_necesarios + 31) / 32) * 32);
      // Calcular bloques necesarios
    int bloques_necesarios = (hilos_necesarios + config.threads_per_block - 1) / config.threads_per_block;
    config.max_blocks = std::min(bloques_necesarios, info.max_blocks_per_grid);
    
    std::cout << "=== CONFIGURACIÓN CUDA OPTIMIZADA ===" << std::endl;
    std::cout << "GPU: " << info.nombre << std::endl;
    std::cout << "Datos por hilo: " << config.ventana_datos_por_hilo << std::endl;
    std::cout << "Threads por bloque: " << config.threads_per_block << std::endl;
    std::cout << "Bloques utilizados: " << config.max_blocks << std::endl;
    std::cout << "Total hilos: " << config.max_blocks * config.threads_per_block << std::endl;
    
    return config;
}

ResultadoDeteccionCUDA detectarAnomaliasCUDA(
    const std::vector<Voto>& votos, 
    const ConfiguracionCUDA& config
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    
    ResultadoDeteccionCUDA resultado;
    int total_votos = votos.size();
    
    if (total_votos == 0) {
        std::cout << "No hay votos para procesar." << std::endl;
        return resultado;
    }
    
    // Obtener información de la GPU
    InfoGPU info = obtenerInfoGPU();
    std::cout << "=== INICIANDO DETECCIÓN CUDA ===" << std::endl;
    std::cout << "GPU: " << info.nombre << std::endl;
    std::cout << "Memoria total: " << info.memoria_total / (1024*1024) << " MB" << std::endl;    std::cout << "Memoria libre: " << info.memoria_libre / (1024*1024) << " MB" << std::endl;
    
    // Calcular configuración óptima
    ConfiguracionCUDA config_optima = calcularConfiguracionOptima(total_votos, config.ventana_datos_por_hilo);
    
    // Convertir votos a formato CUDA
    std::vector<VotoCUDA> votos_cuda(total_votos);
    for (int i = 0; i < total_votos; i++) {
        convertirVotoACUDA(votos[i], votos_cuda[i]);
    }
    
    // Alocar memoria en GPU
    VotoCUDA* d_votos;
    CUDA_CHECK(cudaMalloc(&d_votos, total_votos * sizeof(VotoCUDA)));
    CUDA_CHECK(cudaMemcpy(d_votos, votos_cuda.data(), total_votos * sizeof(VotoCUDA), cudaMemcpyHostToDevice));
    
    // Tamaños de hash tables (primos para mejor distribución)
    int hash_size_dni = 50021;
    int hash_size_region_cand = 25013;
    int hash_size_timestamp = 12007;
    
    // Alocar contadores globales
    int *d_contador_dni, *d_contador_region_candidato, *d_contador_timestamp;
    CUDA_CHECK(cudaMalloc(&d_contador_dni, hash_size_dni * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_contador_region_candidato, hash_size_region_cand * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_contador_timestamp, hash_size_timestamp * sizeof(int)));
    
    // Inicializar contadores a cero
    CUDA_CHECK(cudaMemset(d_contador_dni, 0, hash_size_dni * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_contador_region_candidato, 0, hash_size_region_cand * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_contador_timestamp, 0, hash_size_timestamp * sizeof(int)));
    
    // FASE 1: Construir contadores globales
    std::cout << "\n=== FASE 1: Construyendo contadores globales ===" << std::endl;
      dim3 gridSize(config_optima.max_blocks);
    dim3 blockSize(config_optima.threads_per_block);
    
    construirContadoresGlobales<<<gridSize, blockSize>>>(
        d_votos, total_votos,
        d_contador_dni, d_contador_region_candidato, d_contador_timestamp,
        hash_size_dni, hash_size_region_cand, hash_size_timestamp,
        config_optima.ventana_datos_por_hilo
    );
    CUDA_CHECK_KERNEL();
    
    // Copiar contadores de vuelta para calcular umbrales
    std::vector<int> h_contador_region_cand(hash_size_region_cand);
    std::vector<int> h_contador_timestamp(hash_size_timestamp);
    CUDA_CHECK(cudaMemcpy(h_contador_region_cand.data(), d_contador_region_candidato, 
                         hash_size_region_cand * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_contador_timestamp.data(), d_contador_timestamp, 
                         hash_size_timestamp * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Calcular umbrales estadísticos
    std::vector<int> valores_concentracion, valores_flujo;
    for (int i = 0; i < hash_size_region_cand; i++) {
        if (h_contador_region_cand[i] > 0) {
            valores_concentracion.push_back(h_contador_region_cand[i]);
        }
    }
    for (int i = 0; i < hash_size_timestamp; i++) {
        if (h_contador_timestamp[i] > 0) {
            valores_flujo.push_back(h_contador_timestamp[i]);
        }
    }
    
    double media_conc = 0.0, std_conc = 0.0;
    if (!valores_concentracion.empty()) {
        media_conc = std::accumulate(valores_concentracion.begin(), valores_concentracion.end(), 0.0) / valores_concentracion.size();
        double suma_cuadrados = 0.0;
        for (int val : valores_concentracion) {
            suma_cuadrados += (val - media_conc) * (val - media_conc);
        }
        std_conc = std::sqrt(suma_cuadrados / valores_concentracion.size());
    }
    
    double media_flujo = 0.0, std_flujo = 0.0;
    if (!valores_flujo.empty()) {
        media_flujo = std::accumulate(valores_flujo.begin(), valores_flujo.end(), 0.0) / valores_flujo.size();
        double suma_cuadrados = 0.0;
        for (int val : valores_flujo) {
            suma_cuadrados += (val - media_flujo) * (val - media_flujo);
        }
        std_flujo = std::sqrt(suma_cuadrados / valores_flujo.size());
    }
    
    int umbral_concentracion = std::max(static_cast<int>(media_conc + 2.0 * std_conc), 50);
    int umbral_flujo = std::max(static_cast<int>(media_flujo + 2.0 * std_flujo), 100);
    
    std::cout << "Umbrales calculados:" << std::endl;
    std::cout << "- Concentración: " << umbral_concentracion << " (media: " << media_conc << ", std: " << std_conc << ")" << std::endl;
    std::cout << "- Flujo: " << umbral_flujo << " (media: " << media_flujo << ", std: " << std_flujo << ")" << std::endl;
    
    // FASE 2: Detectar anomalías
    std::cout << "\n=== FASE 2: Detectando anomalías ===" << std::endl;
    
    int *d_contador_duplicados, *d_contador_concentracion, *d_contador_flujo;
    CUDA_CHECK(cudaMalloc(&d_contador_duplicados, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_contador_concentracion, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_contador_flujo, sizeof(int)));
    
    CUDA_CHECK(cudaMemset(d_contador_duplicados, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_contador_concentracion, 0, sizeof(int)));    CUDA_CHECK(cudaMemset(d_contador_flujo, 0, sizeof(int)));
    
    detectarAnomalias<<<gridSize, blockSize>>>(
        d_votos, total_votos,
        d_contador_dni, d_contador_region_candidato, d_contador_timestamp,
        hash_size_dni, hash_size_region_cand, hash_size_timestamp,
        umbral_concentracion, umbral_flujo,
        config_optima.ventana_datos_por_hilo,
        d_contador_duplicados, d_contador_concentracion, d_contador_flujo
    );
    CUDA_CHECK_KERNEL();
    
    // Copiar resultados de vuelta
    CUDA_CHECK(cudaMemcpy(votos_cuda.data(), d_votos, total_votos * sizeof(VotoCUDA), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaMemcpy(&resultado.anomalias_duplicados, d_contador_duplicados, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&resultado.anomalias_concentracion, d_contador_concentracion, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&resultado.anomalias_flujo_excesivo, d_contador_flujo, sizeof(int), cudaMemcpyDeviceToHost));
    
    // FASE 3: Consolidar resultados
    std::cout << "\n=== FASE 3: Consolidando resultados ===" << std::endl;
    
    for (int i = 0; i < total_votos; i++) {
        Voto voto_resultado;
        convertirVotoDeCUDA(votos_cuda[i], voto_resultado);
        
        if (voto_resultado.anomalia_detectada) {
            resultado.anomalos.push_back(voto_resultado);
        } else {
            resultado.validos.push_back(voto_resultado);
        }
    }
    
    // FASE 4: Calcular métricas
    int VP = 0, FP = 0, FN = 0, VN = 0;
    
    for (const auto& v : resultado.anomalos) {
        if (v.anomalo && v.anomalia_detectada) {
            VP++;
        } else if (!v.anomalo && v.anomalia_detectada) {
            FP++;
        }
    }
    
    for (const auto& v : resultado.validos) {
        if (v.anomalo && !v.anomalia_detectada) {
            FN++;
        } else if (!v.anomalo && !v.anomalia_detectada) {
            VN++;
        }
    }
    
    resultado.precision = (VP + FP) > 0 ? static_cast<double>(VP) / (VP + FP) : 0.0;
    resultado.recall = (VP + FN) > 0 ? static_cast<double>(VP) / (VP + FN) : 0.0;
    resultado.f1_score = (resultado.precision + resultado.recall) > 0.0 ?                        2.0 * resultado.precision * resultado.recall / (resultado.precision + resultado.recall) : 0.0;
    
    // Información adicional de GPU
    resultado.bloques_utilizados = config_optima.max_blocks;
    resultado.hilos_por_bloque = config_optima.threads_per_block;
    resultado.datos_por_hilo = config_optima.ventana_datos_por_hilo;
    
    size_t memoria_utilizada = total_votos * sizeof(VotoCUDA) + 
                              (hash_size_dni + hash_size_region_cand + hash_size_timestamp) * sizeof(int) +
                              3 * sizeof(int);
    resultado.memoria_gpu_utilizada_mb = memoria_utilizada / (1024.0f * 1024.0f);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    resultado.tiempo_proceso_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // Liberar memoria GPU
    CUDA_CHECK(cudaFree(d_votos));
    CUDA_CHECK(cudaFree(d_contador_dni));
    CUDA_CHECK(cudaFree(d_contador_region_candidato));
    CUDA_CHECK(cudaFree(d_contador_timestamp));
    CUDA_CHECK(cudaFree(d_contador_duplicados));
    CUDA_CHECK(cudaFree(d_contador_concentracion));
    CUDA_CHECK(cudaFree(d_contador_flujo));
    
    // Mostrar resultados
    std::cout << "\n=== RESULTADOS DE DETECCIÓN CUDA ===" << std::endl;
    std::cout << "Total de votos procesados: " << total_votos << std::endl;
    std::cout << "Votos válidos detectados: " << resultado.validos.size() << std::endl;
    std::cout << "Votos anómalos detectados: " << resultado.anomalos.size() << std::endl;
    std::cout << "Anomalías por duplicados: " << resultado.anomalias_duplicados << std::endl;
    std::cout << "Anomalías por concentración: " << resultado.anomalias_concentracion << std::endl;
    std::cout << "Anomalías por flujo excesivo: " << resultado.anomalias_flujo_excesivo << std::endl;
    std::cout << "\n=== MÉTRICAS ESTADÍSTICAS ===" << std::endl;
    std::cout << "VP (Verdaderos Positivos): " << VP << std::endl;
    std::cout << "FP (Falsos Positivos): " << FP << std::endl;
    std::cout << "FN (Falsos Negativos): " << FN << std::endl;
    std::cout << "VN (Verdaderos Negativos): " << VN << std::endl;
    std::cout << "Precisión: " << resultado.precision << std::endl;
    std::cout << "Recall: " << resultado.recall << std::endl;
    std::cout << "F1-Score: " << resultado.f1_score << std::endl;
    std::cout << "Tiempo de procesamiento: " << resultado.tiempo_proceso_ms << " ms" << std::endl;
    std::cout << "Memoria GPU utilizada: " << resultado.memoria_gpu_utilizada_mb << " MB" << std::endl;
    std::cout << "===============================" << std::endl;
    
    return resultado;
}

void limpiarMemoriaGPU() {
    CUDA_CHECK(cudaDeviceReset());
}

} // namespace deteccion_cuda
