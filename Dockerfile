# =============================================================================
#  Imagen del cluster MPI (nodo maestro + nodos esclavos)
#
#  Compila el sistema C++ y lo deja listo para lanzarse con mpirun. Se usa
#  desde docker-compose.yml, que ademas levanta el dashboard web.
# =============================================================================

# ---- Etapa 1: compilacion ---------------------------------------------------
FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        pkg-config \
        libopenmpi-dev \
        openmpi-bin \
        libcurl4-openssl-dev \
        libjsoncpp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY CMakeLists.txt main.cpp test_mpi.cpp ./
COPY VOTACION ./VOTACION

# La imagen de ejecucion solo necesita el binario del cluster. Ni los tests ni
# el banco de pruebas: sus fuentes ni siquiera se copian a este contexto, asi
# que hay que desactivar ambos objetivos o CMake falla al no encontrarlas.
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF \
    && cmake --build build --parallel


# ---- Etapa 2: ejecucion -----------------------------------------------------
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin \
        libopenmpi3t64 \
        libcurl4t64 \
        libjsoncpp25 \
        libgomp1 \
        python3 \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# OpenMPI se niega a ejecutarse como root salvo que se le fuerce, y forzarlo
# es mala practica. Creamos un usuario sin privilegios para el cluster.
RUN useradd --create-home --shell /bin/bash cluster

WORKDIR /app

COPY --from=build /src/build/votacion /src/build/test_mpi /app/bin/
COPY SCRIPTS ./SCRIPTS
COPY docker/entrypoint-cluster.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh && mkdir -p /app/DATA && chown -R cluster:cluster /app

USER cluster

ENTRYPOINT ["/app/entrypoint.sh"]
