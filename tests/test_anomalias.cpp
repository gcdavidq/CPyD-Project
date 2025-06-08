#include "VOTACION/deteccion/detectar_anomalias.hpp"
#include "VOTACION/common/estructura_votos.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace deteccion;

int main() {
    vector<Voto> votos{
        {"2023-10-01 12:00:00", "Region1", "123456781", "Candidato1", true},
        {"2023-10-01 12:00:01", "Region1", "234567892", "Candidato2", false},
        {"2023-10-01 12:00:02", "Region2", "345678903", "Candidato1", true},
        {"2023-10-01 12:00:03", "Region2", "456789014", "Candidato2", false},
        {"2023-10-01 12:00:04", "Region1", "567890125", "Candidato1", true},
        {"2023-10-01 12:00:05", "Region2", "678901236", "Candidato2", false},
        {"2023-10-01 12:00:06", "Region1", "789012347", "Candidato1", true},
        {"2023-10-01 12:00:07", "Region2", "890123458", "Candidato2", false}
    };

#ifdef USE_CUDA
    // GPU (CUDA)
    cout << "=== Detectando anomalías con CUDA ===" << endl;
    vector<bool> anomalias_reales;
    for (const auto& voto : votos) {
        anomalias_reales.push_back(voto.anomalo);
    }
    vector<bool> anomalias_detectadas;
    detectarAnomaliasCUDA(anomalias_reales, anomalias_detectadas);

    for (size_t i = 0; i < anomalias_detectadas.size(); ++i) {
        cout << votos[i].dni << " - Real: " << anomalias_reales[i]
             << ", Detectada: " << anomalias_detectadas[i] << endl;
    }

#else
    // CPU
    cout << "=== Detectando anomalías con CPU ===" << endl;
    ResultadoDeteccion R = detectarAnomaliasCPU(votos);
    for (const auto& voto : R.validos) {
        cout << voto.dni << " - Anomalo: " << voto.anomalo
             << ", Anomalia detectada: " << voto.anomalia_detectada << endl;
    }
    for (const auto& voto : R.anomalos) {
        cout << voto.dni << " - Anomalo: " << voto.anomalo
             << ", Anomalia detectada: " << voto.anomalia_detectada << endl;
    }
#endif

    return 0;
}
