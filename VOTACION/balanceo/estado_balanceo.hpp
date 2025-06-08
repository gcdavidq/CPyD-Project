/*
SCRIPT 
*/

#pragma once //para evitar inclusiones múltiples
#include <vector>
#include <map>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "VOTACION/common/estructura_votos.hpp"

struct EstadoBalanceo {
    std::vector<RendimientoNodo> rendimiento_nodos;
    std::map<int,int> reasignaciones; // <nodo origen, cantidad_reasignada>
    int lotes_balanceados;
    std::chrono::time_point<std::chrono::system_clock> timestamp; // Marca de tiempo del balanceo

    void imprimirGraficoBalanceo();
    


};