/*
Script para simular la llegada de votos desde un archivo CSV.
*/
#include "VOTACION/simulacion/simulacion_llegada.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include <thread>
#include <chrono> 
#include <fstream>
#include <algorithm>
#include <cctype>
#include <mutex>



static std::string trim(const std::string& s) {
    // opcional: elimina espacios en ambos extremos
    auto start = s.find_first_not_of(" \t\r\n");
    auto end   = s.find_last_not_of(" \t\r\n");
    return (start == std::string::npos)
         ? std::string{}
         : s.substr(start, end - start + 1);
}

std::vector<Voto> leerVotos(const std::string& ruta)
{
    std::vector<Voto> votos;
    std::ifstream file(ruta);
    if (!file.is_open()) {
        std::cerr << "Error al abrir " << ruta << '\n';
        return votos;
    }

    int mostrados   = 0;
    const int maxMostrar = 20;
    std::string linea;

    // Saltar encabezado
    std::getline(file, linea);

    while (std::getline(file, linea)) {
        std::stringstream ss(linea);
        std::string campo;
        Voto v;

        // Leemos los primeros cuatro campos directamente
        std::getline(ss, v.timestamp, ',');
        std::getline(ss, v.region,    ',');
        std::getline(ss, v.dni,       ',');
        std::getline(ss, v.candidato, ',');

        // Leemos el campo "anomalo"
        std::getline(ss, campo, ',');

        // 1) Eliminamos todos los '\r' que pudieran quedar
        campo.erase(std::remove(campo.begin(), campo.end(), '\r'), campo.end());

        // 2) Opcional: trim de espacios en los extremos
        campo = trim(campo);

        // Ahora la comparación funciona correctamente
        v.anomalo = (campo == "1");
        

        /*
        // Mostrar hasta los primeros maxMostrar anomalos
        if (v.anomalo && mostrados < maxMostrar) {
            std::cout << "[ANÓMALO] "
                      << v.timestamp << ", "
                      << v.region    << ", "
                      << v.dni       << ", "
                      << v.candidato << '\n';
            ++mostrados;
        }*/

        votos.push_back(std::move(v));
    }
    /*

    if (mostrados == 0) {
        std::cout << "NO HAY VOTOS ANÓMALOS\n";
    }*/
    return votos;
}
/*
// Versión de diagnóstico para entender el problema
std::vector<Voto> leerVotosDebug(const std::string& ruta)
{
    std::vector<Voto> votos;
    std::ifstream file(ruta);
    if (!file.is_open()) {
        std::cerr << "Error al abrir " << ruta << '\n';
        return votos;
    }
    
    std::cout << "=== INICIANDO DIAGNÓSTICO PARA: " << ruta << " ===" << std::endl;
    
    int mostrados = 0;
    int max_mostrar = 20;
    int linea_num = 0;
    
    std::string linea; 
    getline(file, linea); // encabezado
    std::cout << "Encabezado: " << linea << std::endl;
    
    while (getline(file, linea) && linea_num < 10) { // Solo primeras 10 líneas para debug
        linea_num++;
        std::cout << "\n--- Línea " << linea_num << " ---" << std::endl;
        std::cout << "Contenido completo: '" << linea << "'" << std::endl;
        std::cout << "Longitud: " << linea.length() << std::endl;
        
        // Mostrar cada carácter al final
        if (!linea.empty()) {
            std::cout << "Últimos caracteres: ";
            for (int i = std::max(0, (int)linea.length() - 5); i < linea.length(); i++) {
                std::cout << "'" << linea[i] << "'(" << (int)linea[i] << ") ";
            }
            std::cout << std::endl;
        }
        
        std::stringstream ss(linea);
        std::string campo;
        std::vector<std::string> campos;
        
        // Separar todos los campos
        while (getline(ss, campo, ',')) {
            campos.push_back(campo);
        }
        
        std::cout << "Campos encontrados: " << campos.size() << std::endl;
        for (int i = 0; i < campos.size(); i++) {
            std::cout << "  Campo " << i << ": '" << campos[i] << "' (len: " << campos[i].length() << ")" << std::endl;
        }
        
        if (campos.size() >= 5) {
            std::string anomalo_campo = campos[4];
            std::cout << "Campo anomalo: '" << anomalo_campo << "'" << std::endl;
            std::cout << "¿Es '1'?: " << (anomalo_campo == "1" ? "SÍ" : "NO") << std::endl;
            
            // Verificar cada carácter del campo anomalo
            std::cout << "Caracteres del campo anomalo: ";
            for (char c : anomalo_campo) {
                std::cout << "'" << c << "'(" << (int)c << ") ";
            }
            std::cout << std::endl;
        }
    }
    
    std::cout << "=== FIN DIAGNÓSTICO ===" << std::endl;
    return votos;
} */

void simularLlegadaVotos(const std::vector<Voto>& todos_votos,
                         int nodo_id,
                         std::function<void(std::vector<Voto>&&)> cb,
                         size_t tam_lote)
{
    std::random_device rd; 
    std::mt19937 gen(rd());
    //Distribución aleatoria unforme
    std::uniform_int_distribution<> dist(6000, 10000);

    for (size_t i = 0; i < todos_votos.size(); i += tam_lote) {
        //Determinar cuantos votos procesar en este lote
        std::size_t fin = std::min(i + tam_lote, todos_votos.size());

        //Crear un lote con los votos del rango [i, fin)
        std::vector<Voto> lote(todos_votos.begin() + i, todos_votos.begin() + fin);

        int count_anom = 0;
        for (const auto& v : lote) {
            if (v.anomalo) count_anom++;
        }

        std::cout << "[DEBUG] Lote de " << lote.size() << " votos. Anómalos: " << count_anom << std::endl;

        std::cout << "Nodo " << nodo_id << ": llegaron "
                  << lote.size() << " votos\n";
        

       
        cb(move(lote)); // Invocar el callback con el lote

        std::this_thread::sleep_for(std::chrono::milliseconds(dist(gen)));
    }
}
void simularLlegadaVotos(const std::string& ruta_csv,
                         int nodo_id,
                         //pasamos por referencia el mismo vector, no una copia
                         std::vector<Voto>& votos_recibidos,std::mutex& mtx_votos)
{
    std::vector<Voto> todos_votos = leerVotos(ruta_csv);
    // NUEVO DEBUG
    int total_anom = 0;
    for (const auto& v : todos_votos) {
        if (v.anomalo) total_anom++;
    }
    std::cout << "[DEBUG] Total votos leídos: " << todos_votos.size()
            << " | Anómalos: " << total_anom << std::endl;




    //funcion lambda que inserta los votos del lote a votos_recibidos
    auto cb = [&](std::vector<Voto>&& lote) {
        std::lock_guard<std::mutex> lock(mtx_votos); // <- proteger escritura
        int count_anom = 0;
        for (const auto& v : lote) {
            if (v.anomalo) count_anom++;
        }
        std::cout << "[DEBUG] Insertando lote con " << count_anom << " votos anómalos\n";

        votos_recibidos.insert(
            
            votos_recibidos.end(),
            make_move_iterator(lote.begin()),
            make_move_iterator(lote.end()));

    };

    simularLlegadaVotos(todos_votos, nodo_id, cb, TAM_LOTE_POR_DEFECTO);
}

