// Tests del detector de anomalias en CPU (OpenMP).
//
// Es el nucleo del proyecto y ademas la parte paralela, asi que se comprueban
// dos cosas distintas:
//   1. que detecta lo que tiene que detectar,
//   2. que el resultado NO depende del numero de hilos, que es la garantia de
//      que no hay condiciones de carrera en la region paralela.
#include "doctest.h"
#include "VOTACION/deteccion/detectar_anomalias.hpp"
#include "VOTACION/common/estructura_votos.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace {

Voto voto(const std::string& ts, const std::string& region,
          const std::string& dni, const std::string& candidato,
          bool anomalo = false) {
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

/// Genera votos normales, repartidos entre minutos y candidatos para que
/// ninguna heuristica salte por accidente.
std::vector<Voto> votosNormales(int cantidad) {
    const char* candidatos[] = {"APP", "APRA", "FUERZA POPULAR", "PERU LIBRE"};
    const char* regiones[]   = {"Piura", "Lima", "Cusco", "Ancash"};

    std::vector<Voto> votos;
    votos.reserve(cantidad);
    for (int i = 0; i < cantidad; ++i) {
        char ts[32];
        std::snprintf(ts, sizeof(ts), "2025-04-06 %02d:%02d:00", 8 + (i / 60) % 8, i % 60);
        char dni[16];
        std::snprintf(dni, sizeof(dni), "%08d", 10000000 + i);
        votos.push_back(voto(ts, regiones[i % 4], dni, candidatos[i % 4]));
    }
    return votos;
}

/// Recoge los DNI marcados como anomalos, ordenados, para poder comparar
/// resultados entre ejecuciones.
std::vector<std::string> dnisAnomalos(const deteccion::ResultadoDeteccion& r) {
    std::vector<std::string> dnis;
    for (const auto& v : r.anomalos) dnis.push_back(v.dni);
    std::sort(dnis.begin(), dnis.end());
    return dnis;
}

} // namespace


TEST_CASE("deteccion: ningun voto se pierde ni se duplica") {
    const auto votos = votosNormales(400);
    const auto r = deteccion::detectarAnomaliasCPU(votos, 4);

    CHECK(r.validos.size() + r.anomalos.size() == votos.size());
}

TEST_CASE("deteccion: un DNI repetido se marca como anomalo") {
    auto votos = votosNormales(300);
    // Mismo DNI votando tres veces: es la anomalia mas evidente.
    votos.push_back(voto("2025-04-06 09:15:00", "Piura", "99999999", "APP", true));
    votos.push_back(voto("2025-04-06 10:22:00", "Piura", "99999999", "APP", true));
    votos.push_back(voto("2025-04-06 11:47:00", "Piura", "99999999", "APP", true));

    const auto r = deteccion::detectarAnomaliasCPU(votos, 4);

    const auto marcados = dnisAnomalos(r);
    const auto repetidos = std::count(marcados.begin(), marcados.end(), std::string("99999999"));

    CHECK(repetidos == 3);
    CHECK(r.anomalias_duplicados >= 3);
}

TEST_CASE("deteccion: sin duplicados no se reporta ninguna anomalia por DNI") {
    const auto votos = votosNormales(300);
    const auto r = deteccion::detectarAnomaliasCPU(votos, 4);

    CHECK(r.anomalias_duplicados == 0);
}

TEST_CASE("deteccion: el resultado no depende del numero de hilos") {
    // Esta es la prueba que protege la region paralela. Antes se accedia a los
    // contadores globales con operator[], que puede insertar y rehashear el
    // mapa; con varios hilos eso es una condicion de carrera y el resultado
    // podia variar entre ejecuciones.
    auto votos = votosNormales(1200);
    votos.push_back(voto("2025-04-06 09:00:00", "Lima", "88888888", "APRA", true));
    votos.push_back(voto("2025-04-06 09:00:00", "Lima", "88888888", "APRA", true));

    const auto con1 = deteccion::detectarAnomaliasCPU(votos, 1);
    const auto con2 = deteccion::detectarAnomaliasCPU(votos, 2);
    const auto con4 = deteccion::detectarAnomaliasCPU(votos, 4);
    const auto con8 = deteccion::detectarAnomaliasCPU(votos, 8);

    CHECK(dnisAnomalos(con1) == dnisAnomalos(con2));
    CHECK(dnisAnomalos(con1) == dnisAnomalos(con4));
    CHECK(dnisAnomalos(con1) == dnisAnomalos(con8));

    CHECK(con1.anomalias_duplicados    == con8.anomalias_duplicados);
    CHECK(con1.anomalias_concentracion == con8.anomalias_concentracion);
    CHECK(con1.anomalias_flujo_excesivo == con8.anomalias_flujo_excesivo);
    CHECK(con1.validos.size()          == con8.validos.size());
}

TEST_CASE("deteccion: ejecutar dos veces sobre la misma entrada da lo mismo") {
    const auto votos = votosNormales(600);

    const auto primera  = deteccion::detectarAnomaliasCPU(votos, 4);
    const auto segunda  = deteccion::detectarAnomaliasCPU(votos, 4);

    CHECK(dnisAnomalos(primera) == dnisAnomalos(segunda));
    CHECK(primera.validos.size() == segunda.validos.size());
}

TEST_CASE("deteccion: las metricas se mantienen en el rango valido") {
    auto votos = votosNormales(500);
    for (int i = 0; i < 20; ++i) {
        votos.push_back(voto("2025-04-06 12:00:00", "Lima", "77777777", "APP", true));
    }

    const auto r = deteccion::detectarAnomaliasCPU(votos, 4);

    CHECK(r.precision >= 0.0);
    CHECK(r.precision <= 1.0);
    CHECK(r.recall    >= 0.0);
    CHECK(r.recall    <= 1.0);
    CHECK(r.f1_score  >= 0.0);
    CHECK(r.f1_score  <= 1.0);
    CHECK(r.tiempo_proceso_ms >= 0.0);
}

TEST_CASE("deteccion: una entrada vacia no revienta") {
    const std::vector<Voto> vacio;
    const auto r = deteccion::detectarAnomaliasCPU(vacio, 4);

    CHECK(r.validos.empty());
    CHECK(r.anomalos.empty());
    CHECK(r.anomalias_duplicados == 0);
}

TEST_CASE("deteccion: pedir mas hilos de los disponibles no falla") {
    const auto votos = votosNormales(200);
    const auto r = deteccion::detectarAnomaliasCPU(votos, 9999);

    CHECK(r.validos.size() + r.anomalos.size() == votos.size());
}
