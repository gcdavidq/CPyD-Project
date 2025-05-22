/*

*/
#include "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/VOTACION/simulacion/simulacion_llegada.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <thread>
#include <chrono>

using namespace std;

std::vector<Voto> leerVotos(const string& ruta)
{
    vector<Voto> votos;
    ifstream file(ruta);
    if (!file.is_open()) {
        cerr << "Error al abrir " << ruta << '\n';
        return votos;
    }
    string linea; getline(file, linea);               // encabezado
    while (getline(file, linea)) {
        stringstream ss(linea);
        string campo;
        Voto v;
        getline(ss, v.timestamp, ',');
        getline(ss, v.region,    ',');
        getline(ss, v.dni,       ',');
        getline(ss, v.candidato, ',');
        getline(ss, campo,       ',');
        v.anomalo = (campo == "1");
        votos.push_back(std::move(v));
    }
    return votos;
}

void simularLlegadaVotos(const vector<Voto>& votos,
                         int nodo_id,
                         function<void(vector<Voto>&&)> cb,
                         size_t tam_lote)
{
    random_device rd; 
    mt19937 gen(rd());
    uniform_int_distribution<> dist(100, 500);

    for (size_t i = 0; i < votos.size(); i += tam_lote) {
        size_t fin = min(i + tam_lote, votos.size());
        vector<Voto> lote(votos.begin() + i, votos.begin() + fin);

        cout << "Nodo " << nodo_id << ": llegaron "
                  << lote.size() << " votos\n";

        cb(move(lote));                               // <<<<<<

        this_thread::sleep_for(chrono::milliseconds(dist(gen)));
    }
}
