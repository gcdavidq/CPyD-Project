#include "estadisticas.hpp"
#include <iostream>
#include <iomanip>

using namespace std;

void imprimirEstadisticas(const Estadisticas& stats, int nodo_id){
    cout << "\n========== ESTADÍSTICAS DE PROCESAMIENTO ";
    if (nodo_id >=0) cout <<"NODO" << nodo_id;
    cout <<"======================\n";

    cout<<"Total votos procesados: " << stats.total_votos << endl;
    cout<<"Anomalías reales: " << stats.anomalias_reales << endl;
    cout<<"Anomalías detectadas: " << stats.anomalias_detectadas << endl;
    cout<<"Falsos positivos: " << stats.falsos_positivos << endl;
    cout<<"Falsos negativos: " << stats.falsos_negativos << endl;

    if (stats.anomalias_reales > 0){
        float precision = 100.0f * (stats.anomalias_detectadas - stats.falsos_positivos) / stats.anomalias_detectadas;
        float recall = 100.0f * (stats.anomalias_reales - stats.falsos_negativos) / stats.anomalias_reales;
        cout << "Precisión: " << fixed << setprecision(2) << precision << "%" << endl;
        cout << "Recall: " << fixed << setprecision(2) << recall << "%" << endl;
    }

    cout << "\nVotos por región:\n";
    for (const auto& par : stats.votos_por_region) {
        cout << "  " << par.first << ": " << par.second << endl;
    }

    cout << "\nVotos por candidato:\n";
    for (const auto& par : stats.votos_por_candidato) {
        cout << "  " << par.first << ": " << par.second << endl;
    }

    cout << "================================================\n";
}