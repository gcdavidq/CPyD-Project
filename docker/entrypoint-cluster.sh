#!/usr/bin/env bash
#
# Arranque del cluster MPI dentro del contenedor:
#   1. genera los CSV de votos si todavia no existen,
#   2. espera a que el dashboard responda,
#   3. lanza mpirun con 1 maestro + N esclavos.
#
set -euo pipefail

NUM_ESCLAVOS="${NUM_ESCLAVOS:-4}"
ESCALA_DATOS="${ESCALA_DATOS:-10}"
DASHBOARD_URL="${VOTACION_DASHBOARD_URL:-http://dashboard:5000}"
NUM_HILOS="${VOTACION_NUM_HILOS:-4}"
DIRECTORIO_DATOS="${DIRECTORIO_DATOS:-/app/DATA}"

export VOTACION_DASHBOARD_URL="$DASHBOARD_URL"
export VOTACION_DATA_PREFIX="${VOTACION_DATA_PREFIX:-$DIRECTORIO_DATOS/votos_region}"

echo "=============================================="
echo " Cluster de votacion distribuida"
echo "   Nodos esclavos : $NUM_ESCLAVOS"
echo "   Escala datos   : ${ESCALA_DATOS}%"
echo "   Hilos por nodo : $NUM_HILOS"
echo "   Dashboard      : $DASHBOARD_URL"
echo "=============================================="

# --- 1. Datos ---------------------------------------------------------------
# Cada nodo esclavo de rank N lee "<prefijo>N.csv", por lo que hacen falta
# tantos ficheros como esclavos.
faltan=0
for i in $(seq 1 "$NUM_ESCLAVOS"); do
    if [ ! -f "${DIRECTORIO_DATOS}/votos_region${i}.csv" ]; then
        faltan=1
    fi
done

if [ "$faltan" -eq 1 ]; then
    echo "[1/3] Generando datos de votacion (escala ${ESCALA_DATOS}%)..."
    python3 SCRIPTS/generador_votos.py \
        --todas \
        --escala "$ESCALA_DATOS" \
        --salida "$DIRECTORIO_DATOS" \
        --semilla 42
else
    echo "[1/3] Los datos ya existen, se reutilizan."
fi

# --- 2. Dashboard -----------------------------------------------------------
echo "[2/3] Esperando al dashboard en $DASHBOARD_URL ..."
for intento in $(seq 1 30); do
    if python3 -c "import urllib.request,sys; urllib.request.urlopen('${DASHBOARD_URL}/health', timeout=2)" 2>/dev/null; then
        echo "      Dashboard disponible."
        break
    fi
    if [ "$intento" -eq 30 ]; then
        echo "      AVISO: el dashboard no respondio. El cluster seguira igualmente;"
        echo "      los resultados se veran solo por consola."
    fi
    sleep 2
done

# --- 3. Cluster -------------------------------------------------------------
# Un proceso maestro (rank 0) mas NUM_ESCLAVOS procesos trabajadores.
TOTAL_PROCESOS=$((NUM_ESCLAVOS + 1))

echo "[3/3] Lanzando mpirun con $TOTAL_PROCESOS procesos (1 maestro + $NUM_ESCLAVOS esclavos)..."
echo ""

# --oversubscribe permite mas procesos MPI que nucleos fisicos, util cuando se
# prueba un cluster de 4 nodos en un portatil o en un contenedor limitado.
exec mpirun \
    --oversubscribe \
    -np "$TOTAL_PROCESOS" \
    /app/bin/votacion "$NUM_HILOS"
