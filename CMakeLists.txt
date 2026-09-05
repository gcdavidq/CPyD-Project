# =============================================================================
#  Sistema de Votacion Electronica Distribuida con Deteccion de Anomalias
#  Build: MPI (entre nodos) + OpenMP (dentro de cada nodo) + CUDA (opcional)
# =============================================================================
cmake_minimum_required(VERSION 3.16)
project(VotacionDistribuida VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de build" FORCE)
endif()

option(USE_CUDA          "Compilar el detector de anomalias acelerado por GPU" OFF)
option(BUILD_TESTING     "Compilar la bateria de tests"                        ON)
option(BUILD_BENCHMARKS  "Compilar el banco de pruebas de rendimiento"         ON)

# --- Dependencias ------------------------------------------------------------
find_package(MPI    REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CURL   REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP REQUIRED IMPORTED_TARGET jsoncpp)

# --- Libreria con toda la logica del sistema ---------------------------------
# Se compila una sola vez y la comparten el ejecutable y los tests.
add_library(votacion_core STATIC
  VOTACION/common/config.cpp
  VOTACION/protocolo/protocolo.cpp
  VOTACION/simulacion/simulacion_llegada.cpp
  VOTACION/deteccion/detectar_anomalias.cpp
  VOTACION/procesamiento/procesar_lote.cpp
  VOTACION/estadisticas/estadisticas.cpp
  VOTACION/estadisticas/web_stats_sender.cpp
  VOTACION/balanceo/balanceo_carga.cpp
  VOTACION/balanceo/estado_balanceo.cpp
  VOTACION/rendimiento/rendimiento.cpp
  VOTACION/maestro/nodo_maestro.cpp
  VOTACION/esclavo/nodo_esclavo.cpp
)

# Los #include del proyecto son del tipo "VOTACION/modulo/fichero.hpp",
# por eso la raiz del repositorio tiene que estar en el include path.
target_include_directories(votacion_core PUBLIC ${PROJECT_SOURCE_DIR})

target_link_libraries(votacion_core PUBLIC
  MPI::MPI_CXX
  OpenMP::OpenMP_CXX
  CURL::libcurl
  PkgConfig::JSONCPP
)

# --- Soporte CUDA opcional ---------------------------------------------------
if(USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  target_sources(votacion_core PRIVATE VOTACION/deteccion/detectar_anomalias.cu)
  target_compile_definitions(votacion_core PUBLIC USE_CUDA)
  set_target_properties(votacion_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  message(STATUS "CUDA: ACTIVADO -- se compilara el detector por GPU")
else()
  message(STATUS "CUDA: desactivado -- solo CPU (activalo con -DUSE_CUDA=ON)")
endif()

# --- Ejecutable principal ----------------------------------------------------
# Un solo binario: se comporta como maestro (rank 0) o esclavo (rank > 0).
add_executable(votacion main.cpp)
target_link_libraries(votacion PRIVATE votacion_core)

# --- Utilidad de diagnostico -------------------------------------------------
# Sirve para comprobar que MPI esta bien instalado antes de nada:
#   mpirun -np 4 ./test_mpi
add_executable(test_mpi test_mpi.cpp)
target_link_libraries(test_mpi PRIVATE MPI::MPI_CXX)

# --- Banco de pruebas de rendimiento -----------------------------------------
# Mide el speedup del detector al variar el numero de hilos OpenMP.
if(BUILD_BENCHMARKS)
  add_executable(bench_deteccion benchmarks/bench_deteccion.cpp)
  target_link_libraries(bench_deteccion PRIVATE votacion_core)
endif()

# --- Tests -------------------------------------------------------------------
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# --- Resumen de configuracion ------------------------------------------------
message(STATUS "")
message(STATUS "=========== VotacionDistribuida ${PROJECT_VERSION} ===========")
message(STATUS "  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "  MPI        : ${MPI_CXX_COMPILER}")
message(STATUS "  OpenMP     : ${OpenMP_CXX_FLAGS}")
message(STATUS "  CUDA       : ${USE_CUDA}")
message(STATUS "  Tests      : ${BUILD_TESTING}")
message(STATUS "  Benchmarks : ${BUILD_BENCHMARKS}")
message(STATUS "=================================================")
message(STATUS "")
