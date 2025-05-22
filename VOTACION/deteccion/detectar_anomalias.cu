#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <detectar_anomalias.hpp>
#include <vector>
#include <iostream>

namespace deteccion {

// Kernel CUDA para detectar anomalías (se ejecutará en la GPU)
__global__ void detectarAnomaliasKernel(const bool* datos_entrada, bool* resultados, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Lógica para detectar anomalías
        resultados[idx] = datos_entrada[idx]; // Ejemplo simple: copia directa
    }
}

// Función para llamar al kernel CUDA
void detectarAnomaliasCUDA(const std::vector<bool>& anomalias_reales, std::vector<bool>& anomalias_detectadas) {
    int n = anomalias_reales.size();

    // Asignar memoria en el dispositivo
    bool* d_entrada = nullptr; //nullptr es un puntero nulo
    bool* d_resultados = nullptr;
    cudaMalloc((void**)&d_entrada, n * sizeof(bool));
    cudaMalloc((void**)&d_resultados, n * sizeof(bool));

    // Copiar datos de entrada a la GPU
    cudaMemcpy(d_entrada, anomalias_reales.data(), n * sizeof(bool), cudaMemcpyHostToDevice);

    // Configurar la ejecución
    int threadsPerBlock = 256; // Número de hilos por bloque
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock; 

    // Ejecutar el kernel
    detectarAnomaliasKernel<<<blocksPerGrid, threadsPerBlock>>>(d_entrada, d_resultados, n);

    // Esperar a que la GPU termine
    cudaDeviceSynchronize();

    // Copiar resultados a la CPU
    anomalias_detectadas.resize(n);
    cudaMemcpy(anomalias_detectadas.data(), d_resultados, n * sizeof(bool), cudaMemcpyDeviceToHost);

    // Liberar memoria en la GPU
    cudaFree(d_entrada);
    cudaFree(d_resultados);
}

} // namespace deteccion
#endif
