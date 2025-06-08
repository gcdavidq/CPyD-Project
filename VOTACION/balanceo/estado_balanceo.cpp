#include <VOTACION/balanceo/estado_balanceo.hpp>

void EstadoBalanceo::imprimirGraficoBalanceo() {
    std::cout << "\n========== ESTADO DE BALANCEO DE CARGA ==========\n";
    std::cout << "Timestamp: " << std::chrono::system_clock::to_time_t(timestamp) << "\n\n";

    int max_lotes = 0;
    for (const auto& nodo : rendimiento_nodos) {
        max_lotes = std::max(max_lotes, nodo.lotes_asignados);
    }

    const int ancho_barra = 50;
    for (const auto& nodo : rendimiento_nodos) {
        int longitud_barra = max_lotes > 0 ?
            static_cast<int>((static_cast<float>(nodo.lotes_asignados) / max_lotes) * ancho_barra) : 0;

        std::cout << "Nodo " << nodo.nodo_id << " ";
        if (nodo.tiene_gpu) std::cout << "[GPU] ";
        else std::cout << "      ";

        std::cout << "Tiempo/lote: " << std::fixed << std::setprecision(2)
                  << nodo.tiempo_promedio_lote << "s | ";

        std::cout << "Carga: [";
        for (int i = 0; i < ancho_barra; i++) {
            if (i < longitud_barra) {
                if (nodo.carga_actual > 80) std::cout << "#";
                else if (nodo.carga_actual > 50) std::cout << "+";
                else std::cout << "-";
            } else {
                std::cout << " ";
            }
        }
        std::cout << "] " << nodo.lotes_completados << "/" << nodo.lotes_asignados
                  << " (" << std::fixed << std::setprecision(1) << nodo.carga_actual << "%)\n";
    }

    if (!reasignaciones.empty()) {
        std::cout << "\nReasignaciones en este período:\n";
        for (const auto& par : reasignaciones) {
            std::cout << "  Del nodo " << par.first << " -> " << par.second << " lotes redistribuidos\n";
        }
        std::cout << "  Total: " << lotes_balanceados << " lotes balanceados\n";
    }

    std::cout << "================================================\n\n";


}