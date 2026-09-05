# Tests manuales

Estos dos programas no forman parte de la bateria automatica (`ctest`) porque
necesitan un entorno MPI activo con varios procesos: no se pueden ejecutar como
un test unitario aislado.

- `test_nodo_esclavo.cpp`   arranca un nodo esclavo y observa su ciclo de vida.
- `test_balanceo_carga.cpp` simula escenarios de reparto de carga entre nodos.

Se conservan como herramienta de diagnostico. Para ejecutarlos hay que
compilarlos a mano enlazando con MPI, por ejemplo:

    mpic++ -std=c++17 -fopenmp -I. tests/manual/test_balanceo_carga.cpp \
        VOTACION/balanceo/*.cpp VOTACION/protocolo/*.cpp -o /tmp/test_balanceo
    mpirun -np 4 /tmp/test_balanceo
