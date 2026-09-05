// Tests de la agregacion de estadisticas.
//
// El maestro va combinando los reportes parciales que le llegan de cada nodo
// esclavo. Si combinar() no suma bien, el resultado global es incorrecto sin
// que nada falle de forma visible.
#include "doctest.h"
#include "VOTACION/common/estructura_votos.hpp"

TEST_CASE("estadisticas: combinar suma los contadores escalares") {
    Estadisticas a;
    a.total_votos          = 1000;
    a.anomalias_reales     = 50;
    a.anomalias_detectadas = 45;
    a.falsos_positivos     = 5;
    a.falsos_negativos     = 10;

    Estadisticas b;
    b.total_votos          = 500;
    b.anomalias_reales     = 20;
    b.anomalias_detectadas = 15;
    b.falsos_positivos     = 3;
    b.falsos_negativos     = 2;

    a.combinar(b);

    CHECK(a.total_votos          == 1500);
    CHECK(a.anomalias_reales     == 70);
    CHECK(a.anomalias_detectadas == 60);
    CHECK(a.falsos_positivos     == 8);
    CHECK(a.falsos_negativos     == 12);
}

TEST_CASE("estadisticas: combinar acumula las claves repetidas y anade las nuevas") {
    Estadisticas a;
    a.votos_por_region["Piura"] = 600;
    a.votos_por_region["Lima"]  = 400;
    a.votos_por_candidato["APRA"] = 550;

    Estadisticas b;
    b.votos_por_region["Piura"] = 100;   // ya existia -> debe sumarse
    b.votos_por_region["Cusco"] = 300;   // es nueva   -> debe anadirse
    b.votos_por_candidato["APRA"] = 50;
    b.votos_por_candidato["APP"]  = 200;

    a.combinar(b);

    CHECK(a.votos_por_region.at("Piura") == 700);
    CHECK(a.votos_por_region.at("Lima")  == 400);
    CHECK(a.votos_por_region.at("Cusco") == 300);
    CHECK(a.votos_por_region.size()      == 3);

    CHECK(a.votos_por_candidato.at("APRA") == 600);
    CHECK(a.votos_por_candidato.at("APP")  == 200);
}

TEST_CASE("estadisticas: combinar recorre correctamente los mapas anidados") {
    Estadisticas a;
    a.votos_por_candidato_por_region["Piura"]["APRA"] = 100;
    a.votos_por_candidato_por_region["Piura"]["APP"]  = 50;
    a.anomalias_detectadas_por_region["Piura"]["APRA"] = 7;
    a.anomalias_detectadas_por_candidato["APRA"]["Piura"] = 7;

    Estadisticas b;
    b.votos_por_candidato_por_region["Piura"]["APRA"] = 25;   // suma
    b.votos_por_candidato_por_region["Lima"]["APRA"]  = 900;  // region nueva
    b.anomalias_detectadas_por_region["Piura"]["APRA"] = 3;
    b.anomalias_detectadas_por_candidato["APRA"]["Lima"] = 11;

    a.combinar(b);

    CHECK(a.votos_por_candidato_por_region.at("Piura").at("APRA") == 125);
    CHECK(a.votos_por_candidato_por_region.at("Piura").at("APP")  == 50);
    CHECK(a.votos_por_candidato_por_region.at("Lima").at("APRA")  == 900);

    CHECK(a.anomalias_detectadas_por_region.at("Piura").at("APRA") == 10);
    CHECK(a.anomalias_detectadas_por_candidato.at("APRA").at("Piura") == 7);
    CHECK(a.anomalias_detectadas_por_candidato.at("APRA").at("Lima")  == 11);
}

TEST_CASE("estadisticas: combinar con un reporte vacio no altera nada") {
    Estadisticas a;
    a.total_votos = 1000;
    a.votos_por_region["Piura"] = 600;

    const Estadisticas vacio;
    a.combinar(vacio);

    CHECK(a.total_votos == 1000);
    CHECK(a.votos_por_region.at("Piura") == 600);
    CHECK(a.votos_por_region.size() == 1);
}

TEST_CASE("estadisticas: combinar es asociativo al agregar varios nodos") {
    // El maestro recibe los reportes en orden impredecible, asi que el
    // resultado no puede depender de en que orden se combinen.
    auto hacer = [](int votos, const char* region) {
        Estadisticas e;
        e.total_votos = votos;
        e.votos_por_region[region] = votos;
        return e;
    };

    Estadisticas orden1 = hacer(10, "Piura");
    orden1.combinar(hacer(20, "Lima"));
    orden1.combinar(hacer(30, "Cusco"));

    Estadisticas orden2 = hacer(30, "Cusco");
    orden2.combinar(hacer(10, "Piura"));
    orden2.combinar(hacer(20, "Lima"));

    CHECK(orden1.total_votos      == orden2.total_votos);
    CHECK(orden1.votos_por_region == orden2.votos_por_region);
}
