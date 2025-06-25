/*
Este archivo contiene la declaración de la función detectarAnomaliasCPU y detectarAnomaliasCUDA.
La función detectarAnomaliasCPU se encarga de detectar anomalías en los votos utilizando la CPU, mientras que detectarAnomaliasCUDA se encarga de detectar anomalías utilizando la GPU a través de CUDA.
La función detectarAnomaliasCUDA toma como entrada un vector de booleanos que representan las anomalías reales y devuelve un vector de booleanos que representan las anomalías detectadas.
*/

#pragma once
/**
 * @file detector_anomalias.hpp
 * @brief API pública del módulo de detección de anomalías.
 *
 *  ▸ CPU: usa OpenMP.  
 *  ▸ GPU: usa CUDA (requiere compilar con -DUSE_CUDA).
 */

#include <vector>
#include "VOTACION/common/estructura_votos.hpp"
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <functional>
#include "VOTACION/common/config.hpp"

namespace deteccion {

/** Heurística base empleada (puedes cambiarla por enum o string) */
inline constexpr int MOD_HASH = 3;

/**
 * @brief Resultado compacto de la detección CPU/GPU.
 *
 *  - `validos`         → votos sin anomalía.  
 *  - `anomalos`        → votos marcados como anómalos.  
 *  - `tiempo_proceso`  → milisegundos empleados.
 */
/*  POR AHORA
 struct ResultadoDeteccion {
    std::vector<Voto> validos;
    std::vector<Voto> anomalos;
    double            tiempo_proceso_ms{0.0};
};*/ 
struct ResultadoDeteccion { //NUEVO
    std::vector<Voto> validos;
    std::vector<Voto> anomalos;
    double tiempo_proceso_ms;
    int anomalias_flujo_excesivo;
    int anomalias_concentracion;
    int anomalias_duplicados;
    int anomalias_dnis_falsos;
    double precision;
    double recall;
    double f1_score;
};
struct EstadisticasRegion { //NUEVO
    double media_votos_minuto;
    double desviacion_votos_minuto;
    std::vector<double> distribucion_candidatos;
    int total_votos;
    std::unordered_set<std::string> dnis_vistos;
};

/**
 * @brief Detecta anomalías en un lote de votos usando la CPU.
 * @param votos   Copia (const ref) del lote; no se modifica.
 * @param NUM_HILOS_PARA_ALG
 * @return        ResultadoDeteccion con listas separadas.
 */
ResultadoDeteccion detectarAnomaliasCPU(const std::vector<Voto>& votos, int NUM_HILOS_PARA_ALG);

#ifdef USE_CUDA
/**
 * @brief Detecta anomalías usando la GPU vía CUDA.
 * @param votos   Copia (const ref) del lote; no se modifica.
 * @return        ResultadoDeteccion con listas separadas.
 */
ResultadoDeteccion detectarAnomaliasCUDA(const std::vector<Voto>& votos);
#endif

} // namespace deteccion
