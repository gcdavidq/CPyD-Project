cmake_minimum_required(VERSION 3.14)
project(VotacionParcial CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR})

add_executable(nodo_esclavo
  VOTACION/esclavo/nodo_esclavo.cpp
  VOTACION/simulacion/simulacion_llegada.cpp
  VOTACION/deteccion/detectar_anomalias.cpp
)

add_executable(test_protocolo
  tests/test_protocolo.cpp
  VOTACION/protocolo/protocolo.cpp
  VOTACION/common/estructura_votos.hpp
  VOTACION/estadisticas/estadisticas.cpp
)

add_executable(test_estadisticas
  tests/test_estadisticas.cpp
  VOTACION/estadisticas/estadisticas.cpp
  OTACION/common/estructura_votos.hpp
)
