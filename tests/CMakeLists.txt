# Bateria de tests automatizados (doctest).
#
# Se compila un unico binario con todos los casos y se registra en CTest:
#     cmake --build build && ctest --test-dir build --output-on-failure

add_executable(tests_votacion
  test_main.cpp
  test_protocolo.cpp
  test_estadisticas.cpp
  test_anomalias.cpp
  test_procesar_lote.cpp
  test_simulacion_llegada.cpp
)

target_include_directories(tests_votacion PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
target_link_libraries(tests_votacion PRIVATE votacion_core)

# doctest permite listar y filtrar casos, asi que basta con registrar el binario
# entero y dejar que el informe de fallos venga de el.
add_test(NAME suite_completa COMMAND tests_votacion --reporters=console)

# Los tests de deteccion se ejecutan ademas con distinto numero de hilos: si el
# resultado cambiase, habria una condicion de carrera en la region OpenMP.
add_test(NAME deteccion_1_hilo  COMMAND tests_votacion --test-case=*deteccion*)
set_tests_properties(deteccion_1_hilo  PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")

add_test(NAME deteccion_8_hilos COMMAND tests_votacion --test-case=*deteccion*)
set_tests_properties(deteccion_8_hilos PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=8")
