#include "VOTACION/estadisticas/estadisticas.hpp"
#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/estadisticas/web_stats_sender.hpp"

#include <iostream>
#include <iomanip>
#include <map>
#include <string>

namespace {

/// Division protegida: evita el inf/NaN que aparecia cuando el denominador
/// era 0 (por ejemplo, si no se detectaba ninguna anomalia).
double porcentajeSeguro(double numerador, double denominador) {
    return denominador > 0.0 ? 100.0 * numerador / denominador : 0.0;
}

/// Imprime un mapa "clave: valor" con sangria.
void imprimirMapa(const std::map<std::string, int>& datos) {
    for (const auto& par : datos) {
        std::cout << "  " << par.first << ": " << par.second << "\n";
    }
}

/// Imprime un mapa anidado del tipo "nivel1 -> {nivel2: valor}".
void imprimirMapaAnidado(const std::map<std::string, std::map<std::string, int>>& datos,
                         const std::string& etiqueta_externa,
                         const std::string& etiqueta_interna,
                         const std::string& sufijo = "") {
    for (const auto& externo : datos) {
        std::cout << "  " << etiqueta_externa << ": " << externo.first << "\n";
        for (const auto& interno : externo.second) {
            std::cout << "    " << etiqueta_interna << " " << interno.first
                      << ": " << interno.second << sufijo << "\n";
        }
    }
}

} // namespace


void imprimirEstadisticas(const Estadisticas& stats, int nodo_id) {
    std::cout << "\n========== ESTADISTICAS DE PROCESAMIENTO";
    if (nodo_id >= 0) std::cout << " | NODO " << nodo_id;
    std::cout << " ==========\n";

    std::cout << "Total votos procesados : " << stats.total_votos          << "\n"
              << "Anomalias reales       : " << stats.anomalias_reales     << "\n"
              << "Anomalias detectadas   : " << stats.anomalias_detectadas << "\n"
              << "Falsos positivos       : " << stats.falsos_positivos     << "\n"
              << "Falsos negativos       : " << stats.falsos_negativos     << "\n";

    if (stats.anomalias_reales > 0) {
        // Matriz de confusion a partir de los contadores acumulados:
        //   VP = anomalias reales que si se detectaron
        //   VN = votos normales que no se marcaron
        const int VP = stats.anomalias_reales - stats.falsos_negativos;
        const int VN = (stats.total_votos - stats.anomalias_reales) - stats.falsos_positivos;

        const double precision = porcentajeSeguro(VP, VP + stats.falsos_positivos);
        const double recall    = porcentajeSeguro(VP, stats.anomalias_reales);
        const double accuracy  = porcentajeSeguro(VP + VN, stats.total_votos);
        const double f1        = (precision + recall) > 0.0
                               ? 2.0 * precision * recall / (precision + recall)
                               : 0.0;

        std::cout << std::fixed << std::setprecision(2)
                  << "Precision              : " << precision << "%\n"
                  << "Recall                 : " << recall    << "%\n"
                  << "Accuracy               : " << accuracy  << "%\n"
                  << "F1-Score               : " << f1        << "%\n";
    }

    std::cout << "\nVotos por region:\n";
    imprimirMapa(stats.votos_por_region);

    std::cout << "\nVotos por candidato:\n";
    imprimirMapa(stats.votos_por_candidato);

    std::cout << "\nVotos por candidato dentro de cada region:\n";
    imprimirMapaAnidado(stats.votos_por_candidato_por_region, "Region", "Candidato");

    std::cout << "\nAnomalias detectadas por region y candidato:\n";
    imprimirMapaAnidado(stats.anomalias_detectadas_por_region, "Region", "Candidato", " anomalias");

    std::cout << "\nAnomalias detectadas por candidato y region:\n";
    imprimirMapaAnidado(stats.anomalias_detectadas_por_candidato, "Candidato", "Region", " anomalias");

    std::cout << "================================================\n";
}


void imprimirEstadisticasWeb(const Estadisticas& stats, int nodo_id, WebStatsSender& web_sender) {
    imprimirEstadisticas(stats, nodo_id);

    if (!web_sender.enviarEstadisticas(stats, nodo_id)) {
        std::cout << "[AVISO] No se pudo publicar las estadisticas en el dashboard web\n";
    }
}


void imprimirInfoNodoWeb(int nodo_id, const RendimientoNodo& rendimiento, WebStatsSender& web_sender) {
    std::cout << "\n---- RENDIMIENTO DEL NODO " << nodo_id << " ----\n"
              << std::fixed << std::setprecision(3)
              << "  Tiempo medio por lote : " << rendimiento.tiempo_promedio_lote   << " s\n"
              << "  Lotes completados     : " << rendimiento.lotes_completados      << "\n"
              << "  Carga actual de CPU   : " << rendimiento.carga_actual           << " %\n"
              << "  Hilos empleados       : " << rendimiento.num_hilos              << "\n"
              << "  Tiempo en MPI         : " << rendimiento.tiempo_comunicacion_mpi << " s\n"
              << "  GPU                   : " << (rendimiento.tiene_gpu ? "SI" : "NO") << "\n";

    if (!web_sender.enviarInfoNodo(nodo_id, rendimiento)) {
        std::cout << "[AVISO] No se pudo publicar el rendimiento del nodo en el dashboard web\n";
    }
}
