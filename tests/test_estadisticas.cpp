/*
Script para probar la funcionalidad de imprimir_estadisticas, además de que la función combinar se implementa bien
*/
#include "VOTACION/estadisticas/estadisticas.hpp"
#include <iostream>
#include <iomanip>

using namespace std;
int main(){
    //Crear estadisticas de ejemplo
    Estadisticas stats;
    stats.total_votos = 1000;
    stats.anomalias_reales = 50;
    stats.anomalias_detectadas = 45;
    stats.falsos_positivos = 5;
    stats.falsos_negativos = 5;
    stats.votos_por_region["Region1"] = 600;
    stats.votos_por_region["Region2"] = 400;
    stats.votos_por_candidato["APRA"] = 550;
    stats.votos_por_candidato["Fuerza Popular"] = 450;

    // Imprimir estadísticas
    imprimirEstadisticas(stats,1);
    cout << "Estadísticas impresas correctamente." << std::endl;
    // Crear estadísticas adicionales para combinar 
    Estadisticas stats_adicionales;
    stats_adicionales.total_votos = 500;
    stats_adicionales.anomalias_reales = 20;
    stats_adicionales.anomalias_detectadas = 15;
    stats_adicionales.falsos_positivos = 3;
    stats_adicionales.falsos_negativos = 2;
    stats_adicionales.votos_por_region["Region3"] = 300;
    stats_adicionales.votos_por_region["Region4"] = 200;
    
    // Combinar estadísticas
    stats.combinar(stats_adicionales);
    cout << "Estadísticas combinadas correctamente." << endl;

    //Imprimir estadisticas combinadas
    imprimirEstadisticas(stats,1);

    return 0;

}