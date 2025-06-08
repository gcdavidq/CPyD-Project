#include "VOTACION/procesamiento/procesar_lote.hpp"
#include "VOTACION/deteccion/detectar_anomalias.hpp"

Estadisticas procesarLote(LoteTrabajo& lote, bool tiene_gpu) {
    Estadisticas stats;
    if (lote.votos.empty()) return stats;

    std::vector<bool> anomalias_reales;
    for (const auto& voto : lote.votos) {
        anomalias_reales.push_back(voto.anomalo);
    }

    if (tiene_gpu) {
#ifdef USE_CUDA
        std::vector<bool> anomalias_detectadas;
        deteccion::detectarAnomaliasCUDA(anomalias_reales, anomalias_detectadas);
        for (size_t i = 0; i < lote.votos.size(); ++i) {
            lote.votos[i].anomalia_detectada = anomalias_detectadas[i];
        }
#else
        deteccion::detectarAnomaliasCPU(lote.votos);
#endif
    } else {
        deteccion::detectarAnomaliasCPU(lote.votos);
    }

    stats.total_votos = lote.votos.size();
    for (const auto& voto : lote.votos) {
        stats.votos_por_region[voto.region]++;
        stats.votos_por_candidato[voto.candidato]++;
        if (voto.anomalo) {
            stats.anomalias_reales++;
            if (voto.anomalia_detectada)
                stats.anomalias_detectadas++;
            else
                stats.falsos_negativos++;
        } else if (voto.anomalia_detectada) {
            stats.falsos_positivos++;
            stats.anomalias_detectadas++;
        }
    }

    return stats;
}
