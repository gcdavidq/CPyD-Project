// Tests de la serializacion binaria que viaja por MPI.
//
// Es el punto mas delicado del sistema: se empaquetan structs con strings de
// longitud variable y mapas anidados a mano, sobre un buffer de char. Si el
// ida y vuelta no es exacto, los nodos intercambian basura silenciosamente.
#include "doctest.h"
#include "VOTACION/protocolo/protocolo.hpp"

#include <string>
#include <vector>

namespace {

Voto hacerVoto(const std::string& ts, const std::string& region,
               const std::string& dni, const std::string& candidato,
               bool anomalo, bool detectada) {
    Voto v;
    v.timestamp = ts;
    v.region = region;
    v.dni = dni;
    v.candidato = candidato;
    v.anomalo = anomalo;
    v.anomalia_detectada = detectada;
    v.tipo_anomalia = -1;
    return v;
}

Voto votoClasificado(const std::string& dni, int tipo) {
    Voto v = hacerVoto("2025-04-06 09:00:00", "Piura", dni, "APP", true, true);
    v.tipo_anomalia = tipo;
    return v;
}

} // namespace


TEST_CASE("protocolo: un lote sobrevive intacto al ida y vuelta") {
    LoteTrabajo original;
    original.id_lote = 42;
    original.inicio_timestamp = "2025-06-05 12:00:00";
    original.fin_timestamp    = "2025-06-05 12:05:00";
    original.votos = {
        hacerVoto("2025-06-05 12:00:01", "Piura",       "70857114", "APRA",           false, false),
        hacerVoto("2025-06-05 12:00:02", "La Libertad", "70857115", "FUERZA POPULAR", false, true),
        hacerVoto("2025-06-05 12:00:03", "Lima",        "70857116", "PERU LIBRE",     true,  true),
    };

    std::vector<char> buffer;
    serializarLote(original, buffer);
    REQUIRE(buffer.size() > 0);

    const LoteTrabajo copia = deserializarLote(buffer);

    CHECK(copia.id_lote           == original.id_lote);
    CHECK(copia.inicio_timestamp  == original.inicio_timestamp);
    CHECK(copia.fin_timestamp     == original.fin_timestamp);
    REQUIRE(copia.votos.size()    == original.votos.size());

    for (size_t i = 0; i < original.votos.size(); ++i) {
        CAPTURE(i);
        CHECK(copia.votos[i].timestamp          == original.votos[i].timestamp);
        CHECK(copia.votos[i].region             == original.votos[i].region);
        CHECK(copia.votos[i].dni                == original.votos[i].dni);
        CHECK(copia.votos[i].candidato          == original.votos[i].candidato);
        CHECK(copia.votos[i].anomalo            == original.votos[i].anomalo);
        CHECK(copia.votos[i].anomalia_detectada == original.votos[i].anomalia_detectada);
    }
}

TEST_CASE("protocolo: un lote vacio no rompe la serializacion") {
    LoteTrabajo vacio;
    vacio.id_lote = 7;
    vacio.inicio_timestamp = "";
    vacio.fin_timestamp    = "";

    std::vector<char> buffer;
    serializarLote(vacio, buffer);

    const LoteTrabajo copia = deserializarLote(buffer);
    CHECK(copia.id_lote == 7);
    CHECK(copia.votos.empty());
}

TEST_CASE("protocolo: las estadisticas conservan sus mapas anidados") {
    Estadisticas original;
    original.total_votos          = 1000;
    original.anomalias_reales     = 50;
    original.anomalias_detectadas = 45;
    original.falsos_positivos     = 5;
    original.falsos_negativos     = 10;

    original.votos_por_region["Piura"]       = 600;
    original.votos_por_region["La Libertad"] = 400;
    original.votos_por_candidato["APRA"]     = 550;
    original.votos_por_candidato["APP"]      = 450;

    original.votos_por_candidato_por_region["Piura"]["APRA"] = 300;
    original.votos_por_candidato_por_region["Piura"]["APP"]  = 300;
    original.votos_por_candidato_por_region["La Libertad"]["APRA"] = 250;

    original.anomalias_detectadas_por_region["Piura"]["APRA"]      = 12;
    original.anomalias_detectadas_por_candidato["APRA"]["Piura"]   = 12;

    std::vector<char> buffer;
    serializarEstadisticas(original, buffer);

    const Estadisticas copia = deserializarEstadisticas(buffer);

    CHECK(copia.total_votos          == 1000);
    CHECK(copia.anomalias_reales     == 50);
    CHECK(copia.anomalias_detectadas == 45);
    CHECK(copia.falsos_positivos     == 5);
    CHECK(copia.falsos_negativos     == 10);

    CHECK(copia.votos_por_region     == original.votos_por_region);
    CHECK(copia.votos_por_candidato  == original.votos_por_candidato);

    // Los mapas de dos niveles son la parte que mas facilmente se descuadra.
    CHECK(copia.votos_por_candidato_por_region      == original.votos_por_candidato_por_region);
    CHECK(copia.anomalias_detectadas_por_region     == original.anomalias_detectadas_por_region);
    CHECK(copia.anomalias_detectadas_por_candidato  == original.anomalias_detectadas_por_candidato);
}

TEST_CASE("protocolo: nombres largos y con espacios no desbordan el buffer") {
    LoteTrabajo lote;
    lote.id_lote = 1;
    lote.inicio_timestamp = "2025-06-05 12:00:00";
    lote.fin_timestamp    = "2025-06-05 12:00:00";
    lote.votos = {
        hacerVoto("2025-06-05 12:00:01", "Madre de Dios", "12345678",
                  "PARTIDO CON UN NOMBRE EXCEPCIONALMENTE LARGO PARA LA PAPELETA",
                  true, false),
    };

    std::vector<char> buffer;
    serializarLote(lote, buffer);
    const LoteTrabajo copia = deserializarLote(buffer);

    REQUIRE(copia.votos.size() == 1);
    CHECK(copia.votos[0].region    == "Madre de Dios");
    CHECK(copia.votos[0].candidato == "PARTIDO CON UN NOMBRE EXCEPCIONALMENTE LARGO PARA LA PAPELETA");
    CHECK(copia.votos[0].anomalo   == true);
}


TEST_CASE("protocolo: el tipo de anomalia viaja con el voto") {
    // Al redistribuir un lote por balanceo de carga, el nodo destino tiene que
    // recibir la clasificacion ya hecha. Antes tipo_anomalia no se serializaba
    // y llegaba como memoria sin inicializar.
    LoteTrabajo lote;
    lote.id_lote = 5;
    lote.inicio_timestamp = "2025-04-06 09:00:00";
    lote.fin_timestamp    = "2025-04-06 09:00:00";
    lote.votos = {
        votoClasificado("11111111", 1),   // DNI duplicado
        votoClasificado("22222222", 2),   // concentracion
        votoClasificado("33333333", 3),   // flujo excesivo
        hacerVoto("2025-04-06 09:00:01", "Lima", "44444444", "APRA", false, false),
    };

    std::vector<char> buffer;
    serializarLote(lote, buffer);
    const LoteTrabajo copia = deserializarLote(buffer);

    REQUIRE(copia.votos.size() == 4);
    CHECK(copia.votos[0].tipo_anomalia == 1);
    CHECK(copia.votos[1].tipo_anomalia == 2);
    CHECK(copia.votos[2].tipo_anomalia == 3);
    CHECK(copia.votos[3].tipo_anomalia == -1);
}

TEST_CASE("protocolo: un Voto recien construido no tiene basura") {
    const Voto v;
    CHECK(v.anomalo            == false);
    CHECK(v.anomalia_detectada == false);
    CHECK(v.tipo_anomalia      == -1);
}
