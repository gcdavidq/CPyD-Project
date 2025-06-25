
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "VOTACION/deteccion/detectar_anomalias.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>

namespace deteccion {

    // -------------------------------
    // CUDA kernel
    // -------------------------------
    __global__ void detectarAnomaliasKernel(char** dnis, bool* resultados, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            char* dni = dnis[idx];
            int len = 0;
            while (dni[len] != '\0') ++len;
            resultados[idx] = (dni[len - 1] == '1');  // ← condición de prueba
        }
    }

    // -------------------------------
    // Función host CUDA
    // -------------------------------
    ResultadoDeteccion detectarAnomaliasCUDA(const std::vector<Voto>& votos) {
        ResultadoDeteccion R;
        int n = votos.size();
        R.validos.reserve(n);
        R.anomalos.reserve(n);

        auto t0 = std::chrono::high_resolution_clock::now();

        // === Preparar strings para GPU ===
        const int max_dni_len = 16;
        std::vector<char> h_buffer(n * max_dni_len, 0);
        std::vector<char*> h_ptrs(n);

        for (int i = 0; i < n; ++i) {
            std::strncpy(&h_buffer[i * max_dni_len], votos[i].dni.c_str(), max_dni_len - 1);
            h_ptrs[i] = &h_buffer[i * max_dni_len];
        }

        // === Reservar memoria GPU ===
        char* d_buffer;
        char** d_ptrs;
        bool* d_resultados;

        cudaMalloc(&d_buffer, h_buffer.size());
        cudaMalloc(&d_ptrs, n * sizeof(char*));
        cudaMalloc(&d_resultados, n * sizeof(bool));

        // === Copiar datos a GPU ===
        cudaMemcpy(d_buffer, h_buffer.data(), h_buffer.size(), cudaMemcpyHostToDevice);

        // Ajustar punteros en GPU
        std::vector<char*> d_ptrs_host(n);
        for (int i = 0; i < n; ++i)
            d_ptrs_host[i] = d_buffer + (i * max_dni_len);
        cudaMemcpy(d_ptrs, d_ptrs_host.data(), n * sizeof(char*), cudaMemcpyHostToDevice);

        // === Ejecutar kernel ===
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        detectarAnomaliasKernel<<<blocksPerGrid, threadsPerBlock>>>(d_ptrs, d_resultados, n);
        cudaDeviceSynchronize();

        // === Recuperar resultados ===
        std::vector<char> h_resultados(n);
        cudaMemcpy(h_resultados.data(), d_resultados, n * sizeof(bool), cudaMemcpyDeviceToHost);

        // === Reconstruir resultado ===
        for (int i = 0; i < n; ++i) {
            Voto v = votos[i];
            v.anomalia_detectada = static_cast<bool>(h_resultados[i]);
            if (v.anomalia_detectada)
                R.anomalos.push_back(v);
            else
                R.validos.push_back(v);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        R.tiempo_proceso_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // === Liberar memoria GPU ===
        cudaFree(d_buffer);
        cudaFree(d_ptrs);
        cudaFree(d_resultados);

        return R;
    }

} // namespace deteccion


#endif

/*

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include "VOTACION/deteccion/detectar_anomalias.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <cstring>
#include <unordered_map>
#include <algorithm>

namespace deteccion {

// Constantes para detección de anomalías
__constant__ float UMBRAL_FLUJO_EXCESIVO = 3.0f;        // Factor multiplicador sobre la media
__constant__ float UMBRAL_CONCENTRACION = 0.6f;        // 60% o más votos para un candidato
__constant__ int VENTANA_TIEMPO_MINUTOS = 5;           // Ventana de análisis temporal
__constant__ int MAX_CANDIDATOS = 10;                  // Máximo número de candidatos
__constant__ int MAX_REGIONES = 20;                    // Máximo número de regiones

// Estructura para datos de voto en GPU
struct VotoGPU {
    int timestamp_minuto;    // Minuto desde inicio de votación
    int region_id;           // ID numérico de región
    long long dni;           // DNI como entero
    int candidato_id;        // ID numérico del candidato
    bool es_anomalo_original; // Flag original del simulador
    int indice_original;     // Índice en el array original
};

// Estructura para estadísticas por ventana temporal
struct EstadisticasVentana {
    int votos_total;
    int votos_por_candidato[MAX_CANDIDATOS];
    float flujo_promedio;
    int region_id;
    int ventana_inicio;
};

// ------------------------------- 
// KERNELS CUDA
// ------------------------------- 

// Kernel para detectar anomalías de flujo excesivo
__global__ void detectarFlujoExcesivoKernel(
    VotoGPU* votos, 
    bool* anomalias_flujo,
    int* votos_por_minuto_region,
    float* media_por_region,
    int n_votos,
    int n_regiones,
    int minutos_totales
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_votos) {
        VotoGPU voto = votos[idx];
        int region_idx = voto.region_id * minutos_totales + voto.timestamp_minuto;
        
        if (region_idx < n_regiones * minutos_totales) {
            int votos_minuto = votos_por_minuto_region[region_idx];
            float media_region = media_por_region[voto.region_id];
            
            // Detectar si el flujo es excesivo
            anomalias_flujo[idx] = (votos_minuto > media_region * UMBRAL_FLUJO_EXCESIVO);
        }
    }
}

// Kernel para detectar concentración anómala de candidatos
__global__ void detectarConcentracionKernel(
    VotoGPU* votos,
    bool* anomalias_concentracion,
    int* votos_por_candidato_ventana,
    int* total_votos_ventana,
    int n_votos,
    int n_ventanas
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_votos) {
        VotoGPU voto = votos[idx];
        
        // Calcular ventana temporal
        int ventana_id = (voto.timestamp_minuto / VENTANA_TIEMPO_MINUTOS) * MAX_REGIONES + voto.region_id;
        
        if (ventana_id < n_ventanas) {
            int candidato_idx = ventana_id * MAX_CANDIDATOS + voto.candidato_id;
            int total_idx = ventana_id;
            
            if (candidato_idx < n_ventanas * MAX_CANDIDATOS && total_idx < n_ventanas) {
                int votos_candidato = votos_por_candidato_ventana[candidato_idx];
                int total_ventana = total_votos_ventana[total_idx];
                
                // Detectar concentración anómala
                if (total_ventana > 0) {
                    float porcentaje = (float)votos_candidato / (float)total_ventana;
                    anomalias_concentracion[idx] = (porcentaje >= UMBRAL_CONCENTRACION);
                }
            }
        }
    }
}

// Kernel para detectar DNIs duplicados
__global__ void detectarDuplicadosKernel(
    VotoGPU* votos_ordenados,
    bool* anomalias_duplicados,
    int n_votos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_votos) {
        bool es_duplicado = false;
        
        // Verificar con el voto anterior
        if (idx > 0 && votos_ordenados[idx].dni == votos_ordenados[idx-1].dni) {
            es_duplicado = true;
        }
        
        // Verificar con el voto siguiente
        if (idx < n_votos - 1 && votos_ordenados[idx].dni == votos_ordenados[idx+1].dni) {
            es_duplicado = true;
        }
        
        anomalias_duplicados[votos_ordenados[idx].indice_original] = es_duplicado;
    }
}

// Kernel para consolidar resultados finales
__global__ void consolidarResultadosKernel(
    bool* anomalias_flujo,
    bool* anomalias_concentracion, 
    bool* anomalias_duplicados,
    bool* resultado_final,
    int* tipo_anomalia,
    int n_votos
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_votos) {
        bool tiene_anomalia = false;
        int tipo = 0; // 0: normal, 1: flujo, 2: concentración, 3: duplicado, 4: múltiple
        
        int count_anomalias = 0;
        
        if (anomalias_flujo[idx]) {
            tiene_anomalia = true;
            tipo = 1;
            count_anomalias++;
        }
        
        if (anomalias_concentracion[idx]) {
            tiene_anomalia = true;
            if (tipo == 0) tipo = 2;
            else if (tipo != 2) tipo = 4; // múltiple
            count_anomalias++;
        }
        
        if (anomalias_duplicados[idx]) {
            tiene_anomalia = true;
            if (tipo == 0) tipo = 3;
            else if (tipo != 3) tipo = 4; // múltiple
            count_anomalias++;
        }
        
        resultado_final[idx] = tiene_anomalia;
        tipo_anomalia[idx] = tipo;
    }
}

// ------------------------------- 
// FUNCIONES AUXILIARES
// ------------------------------- 

// Función para convertir timestamp a minutos
int timestampAMinutos(const std::string& timestamp, const std::string& fecha_base) {
    // Implementación simplificada - en producción usar parsing completo
    // Extraer hora:minuto del timestamp "YYYY-MM-DD HH:MM:SS"
    if (timestamp.length() >= 19) {
        int hora = std::stoi(timestamp.substr(11, 2));
        int minuto = std::stoi(timestamp.substr(14, 2));
        // Asumir votación de 8:00 a 16:00
        return (hora - 8) * 60 + minuto;
    }
    return 0;
}

// Función para mapear strings a IDs
int obtenerIdRegion(const std::string& region, std::unordered_map<std::string, int>& mapa_regiones) {
    auto it = mapa_regiones.find(region);
    if (it == mapa_regiones.end()) {
        int id = mapa_regiones.size();
        mapa_regiones[region] = id;
        return id;
    }
    return it->second;
}

int obtenerIdCandidato(const std::string& candidato, std::unordered_map<std::string, int>& mapa_candidatos) {
    auto it = mapa_candidatos.find(candidato);
    if (it == mapa_candidatos.end()) {
        int id = mapa_candidatos.size();
        mapa_candidatos[candidato] = id;
        return id;
    }
    return it->second;
}

// ------------------------------- 
// FUNCIÓN PRINCIPAL CUDA
// ------------------------------- 

ResultadoDeteccion detectarAnomaliasCUDA(const std::vector<Voto>& votos) {
    ResultadoDeteccion R;
    int n = votos.size();
    R.validos.reserve(n);
    R.anomalos.reserve(n);
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    if (n == 0) {
        R.tiempo_proceso_ms = 0.0;
        return R;
    }
    
    // === PREPARACIÓN DE DATOS ===
    
    // Mapear strings a IDs para optimizar GPU
    std::unordered_map<std::string, int> mapa_regiones;
    std::unordered_map<std::string, int> mapa_candidatos;
    
    std::vector<VotoGPU> votos_gpu(n);
    
    // Convertir votos a formato GPU
    for (int i = 0; i < n; ++i) {
        votos_gpu[i].timestamp_minuto = timestampAMinutos(votos[i].timestamp, "2025-04-06");
        votos_gpu[i].region_id = obtenerIdRegion(votos[i].region, mapa_regiones);
        votos_gpu[i].dni = std::stoll(votos[i].dni);
        votos_gpu[i].candidato_id = obtenerIdCandidato(votos[i].candidato, mapa_candidatos);
        votos_gpu[i].es_anomalo_original = votos[i].anomalo;
        votos_gpu[i].indice_original = i;
    }
    
    int n_regiones = mapa_regiones.size();
    int n_candidatos = mapa_candidatos.size();
    int minutos_totales = 8 * 60; // 8 horas de votación
    int n_ventanas = (minutos_totales / VENTANA_TIEMPO_MINUTOS + 1) * n_regiones;
    
    // === RESERVAR MEMORIA GPU ===
    
    VotoGPU* d_votos;
    VotoGPU* d_votos_ordenados;
    bool* d_anomalias_flujo;
    bool* d_anomalias_concentracion;
    bool* d_anomalias_duplicados;
    bool* d_resultado_final;
    int* d_tipo_anomalia;
    int* d_votos_por_minuto_region;
    float* d_media_por_region;
    int* d_votos_por_candidato_ventana;
    int* d_total_votos_ventana;
    
    cudaMalloc(&d_votos, n * sizeof(VotoGPU));
    cudaMalloc(&d_votos_ordenados, n * sizeof(VotoGPU));
    cudaMalloc(&d_anomalias_flujo, n * sizeof(bool));
    cudaMalloc(&d_anomalias_concentracion, n * sizeof(bool));
    cudaMalloc(&d_anomalias_duplicados, n * sizeof(bool));
    cudaMalloc(&d_resultado_final, n * sizeof(bool));
    cudaMalloc(&d_tipo_anomalia, n * sizeof(int));
    cudaMalloc(&d_votos_por_minuto_region, n_regiones * minutos_totales * sizeof(int));
    cudaMalloc(&d_media_por_region, n_regiones * sizeof(float));
    cudaMalloc(&d_votos_por_candidato_ventana, n_ventanas * MAX_CANDIDATOS * sizeof(int));
    cudaMalloc(&d_total_votos_ventana, n_ventanas * sizeof(int));
    
    // === COPIAR DATOS A GPU ===
    
    cudaMemcpy(d_votos, votos_gpu.data(), n * sizeof(VotoGPU), cudaMemcpyHostToDevice);
    cudaMemcpy(d_votos_ordenados, votos_gpu.data(), n * sizeof(VotoGPU), cudaMemcpyHostToDevice);
    
    // Inicializar arrays
    cudaMemset(d_votos_por_minuto_region, 0, n_regiones * minutos_totales * sizeof(int));
    cudaMemset(d_media_por_region, 0, n_regiones * sizeof(float));
    cudaMemset(d_votos_por_candidato_ventana, 0, n_ventanas * MAX_CANDIDATOS * sizeof(int));
    cudaMemset(d_total_votos_ventana, 0, n_ventanas * sizeof(int));
    
    // === CALCULAR ESTADÍSTICAS USANDO THRUST ===
    
    // Para simplificar, calculamos estadísticas en CPU y las copiamos
    // En una implementación completa, usaríamos thrust::reduce_by_key
    
    std::vector<int> votos_por_minuto_region(n_regiones * minutos_totales, 0);
    std::vector<float> media_por_region(n_regiones, 0.0f);
    std::vector<int> votos_por_candidato_ventana(n_ventanas * MAX_CANDIDATOS, 0);
    std::vector<int> total_votos_ventana(n_ventanas, 0);
    
    // Calcular estadísticas en CPU (optimizable con thrust)
    for (const auto& voto : votos_gpu) {
        // Votos por minuto-región
        int idx_minuto = voto.region_id * minutos_totales + voto.timestamp_minuto;
        if (idx_minuto >= 0 && idx_minuto < votos_por_minuto_region.size()) {
            votos_por_minuto_region[idx_minuto]++;
        }
        
        // Votos por ventana-candidato
        int ventana_id = (voto.timestamp_minuto / VENTANA_TIEMPO_MINUTOS) * n_regiones + voto.region_id;
        if (ventana_id >= 0 && ventana_id < n_ventanas) {
            int candidato_idx = ventana_id * MAX_CANDIDATOS + voto.candidato_id;
            if (candidato_idx >= 0 && candidato_idx < votos_por_candidato_ventana.size()) {
                votos_por_candidato_ventana[candidato_idx]++;
            }
            if (ventana_id < total_votos_ventana.size()) {
                total_votos_ventana[ventana_id]++;
            }
        }
    }
    
    // Calcular medias por región
    for (int r = 0; r < n_regiones; ++r) {
        int total_votos = 0;
        int minutos_con_votos = 0;
        for (int m = 0; m < minutos_totales; ++m) {
            int votos_minuto = votos_por_minuto_region[r * minutos_totales + m];
            total_votos += votos_minuto;
            if (votos_minuto > 0) minutos_con_votos++;
        }
        if (minutos_con_votos > 0) {
            media_por_region[r] = (float)total_votos / (float)minutos_con_votos;
        }
    }
    
    // Copiar estadísticas a GPU
    cudaMemcpy(d_votos_por_minuto_region, votos_por_minuto_region.data(), 
               votos_por_minuto_region.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_media_por_region, media_por_region.data(), 
               media_por_region.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_votos_por_candidato_ventana, votos_por_candidato_ventana.data(),
               votos_por_candidato_ventana.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_total_votos_ventana, total_votos_ventana.data(),
               total_votos_ventana.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // === EJECUTAR KERNELS ===
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // 1. Detectar flujo excesivo
    detectarFlujoExcesivoKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_votos, d_anomalias_flujo, d_votos_por_minuto_region, 
        d_media_por_region, n, n_regiones, minutos_totales
    );
    
    // 2. Detectar concentración de candidatos
    detectarConcentracionKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_votos, d_anomalias_concentracion, d_votos_por_candidato_ventana,
        d_total_votos_ventana, n, n_ventanas
    );
    
    // 3. Ordenar por DNI para detectar duplicados
    thrust::device_ptr<VotoGPU> thrust_votos(d_votos_ordenados);
    thrust::sort(thrust_votos, thrust_votos + n, 
                 [] __device__ (const VotoGPU& a, const VotoGPU& b) {
                     return a.dni < b.dni;
                 });
    
    // 4. Detectar duplicados
    detectarDuplicadosKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_votos_ordenados, d_anomalias_duplicados, n
    );
    
    // 5. Consolidar resultados
    consolidarResultadosKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_anomalias_flujo, d_anomalias_concentracion, d_anomalias_duplicados,
        d_resultado_final, d_tipo_anomalia, n
    );
    
    cudaDeviceSynchronize();
    
    // === RECUPERAR RESULTADOS ===
    
    std::vector<bool> h_resultados(n);
    std::vector<int> h_tipos_anomalia(n);
    
    cudaMemcpy(h_resultados.data(), d_resultado_final, n * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tipos_anomalia.data(), d_tipo_anomalia, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // === RECONSTRUIR RESULTADO FINAL ===
    
    for (int i = 0; i < n; ++i) {
        Voto v = votos[i];
        v.anomalia_detectada = h_resultados[i];
        v.tipo_anomalia = h_tipos_anomalia[i];
        
        if (v.anomalia_detectada) {
            R.anomalos.push_back(v);
        } else {
            R.validos.push_back(v);
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    R.tiempo_proceso_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // === LIBERAR MEMORIA GPU ===
    
    cudaFree(d_votos);
    cudaFree(d_votos_ordenados);
    cudaFree(d_anomalias_flujo);
    cudaFree(d_anomalias_concentracion);
    cudaFree(d_anomalias_duplicados);
    cudaFree(d_resultado_final);
    cudaFree(d_tipo_anomalia);
    cudaFree(d_votos_por_minuto_region);
    cudaFree(d_media_por_region);
    cudaFree(d_votos_por_candidato_ventana);
    cudaFree(d_total_votos_ventana);
    
    return R;
}

} // namespace deteccion

#endif
*/