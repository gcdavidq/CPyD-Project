/*
Este archivo contiene la declaración de la función detectarAnomaliasCPU y detectarAnomaliasCUDA.
La función detectarAnomaliasCPU se encarga de detectar anomalías en los votos utilizando la CPU, mientras que detectarAnomaliasCUDA se encarga de detectar anomalías utilizando la GPU a través de CUDA.
La función detectarAnomaliasCUDA toma como entrada un vector de booleanos que representan las anomalías reales y devuelve un vector de booleanos que representan las anomalías detectadas.
*/
#pragma once
#include <vector>
#include <VOTACION/common/estructura_votos.hpp>

namespace deteccion{
    void detectarAnomaliasCPU(std::vector<Voto>& votos); 

    #ifdef USE_CUDA
    void detectarAnomaliasCUDA(const std::vector<bool>& anomalias_reales, std::vector<bool>&anomalias_detectadas);

    #endif
}