// Tests de la lectura de los CSV de votos.
//
// El fichero de entrada lo genera un script de Python en Linux o en Windows,
// asi que el parser tiene que sobrevivir a finales de linea CRLF y a espacios
// sobrantes. Un fallo aqui hace que TODOS los votos parezcan no anomalos y
// que las metricas de precision del proyecto salgan sin sentido.
#include "doctest.h"
#include "VOTACION/simulacion/simulacion_llegada.hpp"

#include <cstdio>
#include <fstream>
#include <string>

namespace {

/// Escribe un CSV temporal y devuelve su ruta.
std::string escribirCsv(const std::string& contenido, const std::string& nombre) {
    const std::string ruta = std::string(".") + "/" + nombre;
    std::ofstream out(ruta, std::ios::binary);
    out << contenido;
    out.close();
    return ruta;
}

} // namespace


TEST_CASE("leerVotos: interpreta las columnas del CSV") {
    const std::string ruta = escribirCsv(
        "timestamp,region,dni,candidato,anomalo\n"
        "2025-04-06 08:00:00,Piura,12345678,APP,0\n"
        "2025-04-06 08:00:01,Lima,87654321,APRA,1\n",
        "test_votos_basico.csv");

    const auto votos = leerVotos(ruta);
    std::remove(ruta.c_str());

    REQUIRE(votos.size() == 2);

    CHECK(votos[0].timestamp == "2025-04-06 08:00:00");
    CHECK(votos[0].region    == "Piura");
    CHECK(votos[0].dni       == "12345678");
    CHECK(votos[0].candidato == "APP");
    CHECK(votos[0].anomalo   == false);

    CHECK(votos[1].region  == "Lima");
    CHECK(votos[1].anomalo == true);
}

TEST_CASE("leerVotos: soporta finales de linea de Windows (CRLF)") {
    // Si el '\r' no se limpia, el campo pasa a valer "1\r" en vez de "1" y
    // ningun voto se considera anomalo.
    const std::string ruta = escribirCsv(
        "timestamp,region,dni,candidato,anomalo\r\n"
        "2025-04-06 08:00:00,Piura,12345678,APP,1\r\n"
        "2025-04-06 08:00:01,Lima,87654321,APRA,0\r\n",
        "test_votos_crlf.csv");

    const auto votos = leerVotos(ruta);
    std::remove(ruta.c_str());

    REQUIRE(votos.size() == 2);
    CHECK(votos[0].anomalo == true);
    CHECK(votos[1].anomalo == false);
    CHECK(votos[1].candidato == "APRA");
}

TEST_CASE("leerVotos: tolera espacios alrededor del indicador de anomalia") {
    const std::string ruta = escribirCsv(
        "timestamp,region,dni,candidato,anomalo\n"
        "2025-04-06 08:00:00,Piura,12345678,APP, 1 \n",
        "test_votos_espacios.csv");

    const auto votos = leerVotos(ruta);
    std::remove(ruta.c_str());

    REQUIRE(votos.size() == 1);
    CHECK(votos[0].anomalo == true);
}

TEST_CASE("leerVotos: un fichero inexistente devuelve una lista vacia") {
    const auto votos = leerVotos("no_existe_este_fichero_12345.csv");
    CHECK(votos.empty());
}

TEST_CASE("leerVotos: un CSV con solo cabecera devuelve una lista vacia") {
    const std::string ruta = escribirCsv(
        "timestamp,region,dni,candidato,anomalo\n", "test_votos_vacio.csv");

    const auto votos = leerVotos(ruta);
    std::remove(ruta.c_str());

    CHECK(votos.empty());
}
