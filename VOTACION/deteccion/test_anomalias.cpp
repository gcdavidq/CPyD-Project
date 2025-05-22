// test_anomalias.cpp
#include "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/VOTACION/deteccion/detectar_anomalias.hpp"
#include <iostream>

using namespace std;
int main() {
    std::vector<Voto> lote(1000);
    for (int i = 0; i < 1000; ++i)
        lote[i].dni = "ID" + std::to_string(i);

    auto res = deteccion::detectarAnomaliasCPU(lote);
    cout << "Validos: " << res.validos.size()
              << " | Anomalos: " << res.anomalos.size()
              << " | t=" << res.tiempo_proceso_ms << " ms\n";
}
