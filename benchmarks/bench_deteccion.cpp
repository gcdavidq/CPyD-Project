/*
 * Banco de pruebas del detector de anomalias.
 *
 * Mide como escala detectarAnomaliasCPU() al aumentar el numero de hilos
 * OpenMP y emite los resultados en CSV, listos para graficar.
 *
 *   ./bench_deteccion <fichero.csv> [votos_maximos] [repeticiones]
 *
 * Salida (stdout, formato CSV):
 *   hilos,repeticion,votos,tiempo_ms,speedup,eficiencia
 */
#include "VOTACION/deteccion/detectar_anomalias.hpp"
#include "VOTACION/simulacion/simulacion_llegada.hpp"
#include "VOTACION/common/estructura_votos.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <string>
#include <vector>

namespace {

/// Mediana: mas robusta que la media frente a un pico puntual del sistema.
double mediana(std::vector<double> valores) {
    if (valores.empty()) return 0.0;
    std::sort(valores.begin(), valores.end());
    const size_t n = valores.size();
    return (n % 2 == 1) ? valores[n / 2]
                        : (valores[n / 2 - 1] + valores[n / 2]) / 2.0;
}

} // namespace


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <fichero.csv> [votos_maximos] [repeticiones]\n";
        return 1;
    }

    const std::string ruta = argv[1];
    const size_t max_votos = (argc > 2) ? std::strtoul(argv[2], nullptr, 10) : 300000;
    const int repeticiones = (argc > 3) ? std::atoi(argv[3]) : 3;

    std::cerr << "Cargando " << ruta << " ...\n";
    std::vector<Voto> votos = leerVotos(ruta);
    if (votos.empty()) {
        std::cerr << "ERROR: no se pudo leer ningun voto de " << ruta << "\n";
        return 1;
    }
    if (votos.size() > max_votos) votos.resize(max_votos);

    const int hilos_disponibles = omp_get_max_threads();
    std::cerr << "Votos: " << votos.size()
              << " | Hilos disponibles: " << hilos_disponibles
              << " | Repeticiones: " << repeticiones << "\n\n";

    // Configuraciones de hilos: potencias de 2 hasta el maximo del equipo.
    std::vector<int> configuraciones;
    for (int h = 1; h <= hilos_disponibles; h *= 2) configuraciones.push_back(h);
    if (configuraciones.back() != hilos_disponibles) {
        configuraciones.push_back(hilos_disponibles);
    }

    // Ejecucion en vacio para calentar caches y no penalizar a la primera medida.
    (void)deteccion::detectarAnomaliasCPU(votos, 1);

    std::cout << "hilos,repeticion,votos,tiempo_ms,speedup,eficiencia\n";

    double base_ms = 0.0;

    for (const int hilos : configuraciones) {
        std::vector<double> tiempos;

        for (int r = 0; r < repeticiones; ++r) {
            const auto t0 = std::chrono::high_resolution_clock::now();
            const auto resultado = deteccion::detectarAnomaliasCPU(votos, hilos);
            const auto t1 = std::chrono::high_resolution_clock::now();

            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            tiempos.push_back(ms);

            // Se usa el resultado para que el compilador no pueda eliminarlo.
            if (resultado.validos.size() + resultado.anomalos.size() != votos.size()) {
                std::cerr << "ERROR: el detector perdio votos con " << hilos << " hilos\n";
                return 1;
            }
        }

        const double ms = mediana(tiempos);
        if (hilos == 1) base_ms = ms;

        const double speedup    = (ms > 0.0) ? base_ms / ms : 0.0;
        const double eficiencia = speedup / hilos;

        for (int r = 0; r < repeticiones; ++r) {
            std::cout << hilos << "," << r << "," << votos.size() << ","
                      << std::fixed << std::setprecision(2) << tiempos[r] << ","
                      << std::setprecision(3) << (tiempos[r] > 0 ? base_ms / tiempos[r] : 0.0) << ","
                      << std::setprecision(3) << (tiempos[r] > 0 ? (base_ms / tiempos[r]) / hilos : 0.0)
                      << "\n";
        }

        std::cerr << "  " << std::setw(3) << hilos << " hilos: "
                  << std::fixed << std::setprecision(1) << std::setw(8) << ms << " ms"
                  << "   speedup " << std::setprecision(2) << speedup << "x"
                  << "   eficiencia " << std::setprecision(0) << (eficiencia * 100) << "%\n";
    }

    return 0;
}
