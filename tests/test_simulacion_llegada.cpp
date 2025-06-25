/*
Script para probar la función simularLlegadaVotos desde un archivo CSV de prueba.
*/

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include "VOTACION/simulacion/simulacion_llegada.hpp"
#include "VOTACION/common/estructura_votos.hpp"

using namespace std;

int main() {
    string archivo_csv = "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/DATA/votos_simulados_region1.csv";
    int nodo_id = 1;
    vector<Voto> votos_recibidos;

    // Leer todos los votos una sola vez
    vector<Voto> todos_votos = leerVotos(archivo_csv);

    // Crear una lambda como callback para capturar cada lote recibido
    auto callback = [&](vector<Voto>&& lote) {
        // Agregar los votos del lote al vector principal
        votos_recibidos.insert(votos_recibidos.end(),
                               make_move_iterator(lote.begin()),
                               make_move_iterator(lote.end()));
    };

    // Lanzar la simulación en un hilo
    thread simulador(simularLlegadaVotos, todos_votos, nodo_id, callback, 5000);

    // Esperar a que termine la simulación antes de imprimir
    if (simulador.joinable()) simulador.join();
    /*

    // Mostrar los votos recibidos hasta el momento
    cout << "\n--- Votos recibidos por el nodo " << nodo_id << " ---\n";
    for (const auto& voto : votos_recibidos) {
        cout << "Timestamp: " << voto.timestamp
             << ", Región: " << voto.region
             << ", DNI: " << voto.dni
             << ", Candidato: " << voto.candidato
             << ", Anómalo: " << (voto.anomalo ? "Sí" : "No")
             << endl;
    } 
    */
    cout << "Simulación de llegada de votos finalizada. Total de votos recibidos: "
         << votos_recibidos.size() << endl;


    return 0;
}
