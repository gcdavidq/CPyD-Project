// Tests del procesamiento de un lote completo.
//
// procesarLote() encadena la deteccion de anomalias con la construccion de las
// estadisticas que despues viajan al maestro. Aqui se comprueba que los
// contadores cuadran con la entrada.
#include "doctest.h"
#include "VOTACION/procesamiento/procesar_lote.hpp"
#include "VOTACION/common/estructura_votos.hpp"

#include <string>

namespace {

Voto voto(const std::string& ts, const std::string& region,
          const std::string& dni, const std::string& candidato, bool anomalo) {
    Voto v;
    v.timestamp = ts;
    v.region = region;
    v.dni = dni;
    v.candidato = candidato;
    v.anomalo = anomalo;
    v.anomalia_detectada = false;
    v.tipo_anomalia = -1;
    return v;
}

LoteTrabajo loteDeEjemplo() {
    LoteTrabajo lote;
    lote.id_lote = 1;
    lote.inicio_timestamp = "2025-04-06 12:00:00";
    lote.fin_timestamp    = "2025-04-06 12:01:00";
    lote.votos = {
        voto("2025-04-06 12:00:01", "Piura", "12345678", "APP",   false),
        voto("2025-04-06 12:00:02", "Piura", "87654321", "APRA",  false),
        voto("2025-04-06 12:00:03", "Lima",  "23456789", "APP",   false),
        voto("2025-04-06 12:00:04", "Lima",  "34567890", "APRA",  false),
        voto("2025-04-06 12:00:05", "Cusco", "45678901", "APP",   false),
        voto("2025-04-06 12:00:06", "Cusco", "56789012", "APRA",  false),
    };
    return lote;
}

} // namespace


TEST_CASE("procesarLote: el total coincide con los votos de entrada") {
    LoteTrabajo lote = loteDeEjemplo();
    const size_t esperados = lote.votos.size();

    const Estadisticas stats = procesarLote(lote, /*tiene_gpu=*/false);

    CHECK(stats.total_votos == static_cast<int>(esperados));
    // El lote se reconstruye con validos + anomalos: no puede perder votos.
    CHECK(lote.votos.size() == esperados);
}

TEST_CASE("procesarLote: los votos se reparten bien por region y candidato") {
    LoteTrabajo lote = loteDeEjemplo();
    const Estadisticas stats = procesarLote(lote, false);

    CHECK(stats.votos_por_region.at("Piura") == 2);
    CHECK(stats.votos_por_region.at("Lima")  == 2);
    CHECK(stats.votos_por_region.at("Cusco") == 2);

    CHECK(stats.votos_por_candidato.at("APP")  == 3);
    CHECK(stats.votos_por_candidato.at("APRA") == 3);

    CHECK(stats.votos_por_candidato_por_region.at("Piura").at("APP") == 1);
}

TEST_CASE("procesarLote: un lote vacio devuelve estadisticas en cero") {
    LoteTrabajo vacio;
    vacio.id_lote = 99;

    const Estadisticas stats = procesarLote(vacio, false);

    CHECK(stats.total_votos == 0);
    CHECK(stats.anomalias_reales == 0);
    CHECK(stats.votos_por_region.empty());
}

TEST_CASE("procesarLote: las anomalias reales se contabilizan") {
    LoteTrabajo lote = loteDeEjemplo();
    // Dos votos del mismo DNI y ademas etiquetados como anomalos de origen.
    lote.votos.push_back(voto("2025-04-06 12:00:07", "Piura", "11111111", "APP", true));
    lote.votos.push_back(voto("2025-04-06 12:00:08", "Piura", "11111111", "APP", true));

    const Estadisticas stats = procesarLote(lote, false);

    CHECK(stats.anomalias_reales == 2);
    // La suma de la matriz de confusion no puede exceder el total.
    CHECK(stats.falsos_positivos + stats.falsos_negativos <= stats.total_votos);
}

TEST_CASE("procesarLote: sin GPU cae a la ruta de CPU sin fallar") {
    LoteTrabajo lote = loteDeEjemplo();
    // tiene_gpu=true sin compilar con USE_CUDA debe usar la CPU igualmente.
    const Estadisticas stats = procesarLote(lote, /*tiene_gpu=*/true);

    CHECK(stats.total_votos == 6);
}
