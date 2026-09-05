"""
Dashboard web del Sistema de Votacion Electronica Distribuida.

Funciona en dos modos, controlados por la variable de entorno MODO_DASHBOARD:

  live    (por defecto)  El cluster MPI publica aqui sus estadisticas por HTTP.
                         Si ademas GRABAR_EJECUCION=1, todo lo que llega se
                         guarda en un fichero .jsonl con su marca de tiempo.

  replay                 No hay cluster: se reproduce en bucle una ejecucion
                         real previamente grabada. Es el modo que usa la demo
                         publica, para que el visitante vea datos autenticos
                         sin necesidad de levantar OpenMPI en el servidor.
"""
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit

# --------------------------------------------------------------------------- #
#  Configuracion
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

MODO = os.environ.get("MODO_DASHBOARD", "live").lower()
GRABAR = os.environ.get("GRABAR_EJECUCION", "0") == "1"
FICHERO_DEMO = BASE_DIR / os.environ.get("FICHERO_DEMO", "demo/ejecucion_real.jsonl")
PUERTO = int(os.environ.get("PORT", 5000))
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
PAUSA_ENTRE_CICLOS = float(os.environ.get("REPLAY_PAUSA_SEG", 4))
VELOCIDAD_REPLAY = float(os.environ.get("REPLAY_VELOCIDAD", 1.0))

# El frontend es una SPA de React compilada por Vite a static/dist.
DIST = BASE_DIR / "static" / "dist"

app = Flask(__name__, static_folder="static", static_url_path="/static")

# En produccion la clave DEBE venir del entorno. El valor por defecto solo
# sirve para desarrollo local y no protege nada.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "clave-solo-para-desarrollo-local")

# Modo de concurrencia de Socket.IO. Se usa "threading" por defecto a
# proposito: eventlet y gevent van por detras de las versiones nuevas de Python
# (con Python 3.14, por ejemplo, eventlet ni siquiera importa). Con "threading"
# el transporte cae a long-polling en lugar de WebSocket, lo cual es
# perfectamente suficiente para un refresco cada pocos segundos y evita una
# familia entera de incompatibilidades.
ASYNC_MODE = os.environ.get("SOCKETIO_ASYNC_MODE", "threading")

socketio = SocketIO(app, cors_allowed_origins="*", async_mode=ASYNC_MODE)


# --------------------------------------------------------------------------- #
#  Estado compartido
# --------------------------------------------------------------------------- #
def estado_inicial():
    """Estructura vacia de estadisticas.

    Existe como funcion (y no como literal global) porque /api/reset_stats
    necesita reconstruirla intacta: la version anterior hacia {k: 0 for k},
    que convertia los diccionarios en el entero 0 y rompia el frontend.
    """
    return {
        "total_votos": 0,
        "anomalias_reales": 0,
        "anomalias_detectadas": 0,
        "falsos_positivos": 0,
        "falsos_negativos": 0,
        "votos_por_region": {},
        "votos_por_candidato": {},
        "votos_por_candidato_por_region": {},
        "anomalias_por_region_candidato": {},
        "anomalias_por_candidato_region": {},
        "nodos_activos": 0,
        "ultimo_update": datetime.now().strftime("%H:%M:%S"),
        "tiempo_procesamiento": 0,
    }


_lock = threading.Lock()
estadisticas_globales = estado_inicial()
rendimiento_nodos = {}

# Historial del recuento, para que la gráfica tenga pasado en cuanto se abre la
# página. Antes la serie vivía solo en el cliente: quien llegaba con la
# ejecución ya avanzada veía un panel vacío hasta la siguiente actualización.
MAX_HISTORIAL = 400
historial = []
_inicio_serie = time.time()

_grabacion_lock = threading.Lock()
_inicio_grabacion = None


# --------------------------------------------------------------------------- #
#  Grabacion de ejecuciones reales
# --------------------------------------------------------------------------- #
def grabar_evento(tipo, datos):
    """Anade un evento al fichero de grabacion, con su desplazamiento temporal."""
    global _inicio_grabacion
    if not GRABAR:
        return

    with _grabacion_lock:
        if _inicio_grabacion is None:
            _inicio_grabacion = time.time()
            FICHERO_DEMO.parent.mkdir(parents=True, exist_ok=True)
            FICHERO_DEMO.write_text("", encoding="utf-8")

        evento = {
            "t": round(time.time() - _inicio_grabacion, 3),
            "tipo": tipo,
            "datos": datos,
        }
        with FICHERO_DEMO.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(evento, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
#  Aplicacion de eventos al estado
# --------------------------------------------------------------------------- #
def aplicar_estadisticas(datos):
    global _inicio_serie
    with _lock:
        anterior = estadisticas_globales.get("total_votos", 0)
        estadisticas_globales.update(datos)
        estadisticas_globales["nodos_activos"] = len(rendimiento_nodos)
        estadisticas_globales["ultimo_update"] = datetime.now().strftime("%H:%M:%S")

        total = estadisticas_globales.get("total_votos", 0)
        if total != anterior:
            # Un ciclo de replay reinicia el contador: se empieza una serie
            # nueva en vez de dibujar una caída que no ocurrió.
            if total < anterior:
                historial.clear()
                _inicio_serie = time.time()
            historial.append({
                "t": round(time.time() - _inicio_serie, 2),
                "votos": total,
                "anomalias": estadisticas_globales.get("anomalias_detectadas", 0),
            })
            del historial[:-MAX_HISTORIAL]

        instantanea = dict(estadisticas_globales)
    socketio.emit("estadisticas_update", instantanea)


def aplicar_nodo(datos):
    nodo_id = datos.get("nodo_id")
    if nodo_id is None:
        return
    with _lock:
        rendimiento_nodos[str(nodo_id)] = datos
        estadisticas_globales["nodos_activos"] = len(rendimiento_nodos)
        instantanea = dict(rendimiento_nodos)
    socketio.emit("nodos_update", instantanea)


# --------------------------------------------------------------------------- #
#  Modo replay
# --------------------------------------------------------------------------- #
def cargar_grabacion():
    if not FICHERO_DEMO.exists():
        app.logger.warning("No existe la grabacion %s; el replay no arrancara.", FICHERO_DEMO)
        return []

    eventos = []
    with FICHERO_DEMO.open(encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea:
                continue
            try:
                eventos.append(json.loads(linea))
            except json.JSONDecodeError:
                app.logger.warning("Linea invalida en la grabacion, se omite.")
    eventos.sort(key=lambda e: e.get("t", 0))
    return eventos


def bucle_replay():
    """Reproduce la grabacion respetando los intervalos reales, en bucle."""
    eventos = cargar_grabacion()
    if not eventos:
        return

    app.logger.info("Replay: %d eventos cargados de %s", len(eventos), FICHERO_DEMO.name)

    global _inicio_serie

    while True:
        with _lock:
            estadisticas_globales.clear()
            estadisticas_globales.update(estado_inicial())
            rendimiento_nodos.clear()
            # El historial se reinicia aqui tambien. La deteccion de "el total
            # ha bajado" que hay en aplicar_estadisticas no basta: este bucle
            # pone el contador a cero por su cuenta, sin pasar por ella, asi
            # que la serie del ciclo anterior sobrevivia y la grafica acababa
            # encadenando varias ejecuciones en una sola linea.
            historial.clear()
            _inicio_serie = time.time()

        anterior = 0.0
        for evento in eventos:
            espera = (evento.get("t", 0.0) - anterior) / max(VELOCIDAD_REPLAY, 0.01)
            if espera > 0:
                socketio.sleep(min(espera, 10))
            anterior = evento.get("t", 0.0)

            tipo = evento.get("tipo")
            if tipo == "stats":
                aplicar_estadisticas(evento.get("datos", {}))
            elif tipo == "nodo":
                aplicar_nodo(evento.get("datos", {}))

        socketio.sleep(PAUSA_ENTRE_CICLOS)


# --------------------------------------------------------------------------- #
#  Frontend
# --------------------------------------------------------------------------- #
# Todas estas rutas devuelven la misma SPA; el enrutado se resuelve en el
# cliente con la History API. Se listan de forma explicita (en vez de usar un
# comodin) para que una URL inexistente siga dando 404 de verdad.
RUTAS_SPA = ("/", "/panel", "/nodos", "/mapa", "/dashboard", "/sobre")

MENSAJE_SIN_BUILD = """<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<title>Frontend sin compilar</title>
<style>
  body{font-family:system-ui,-apple-system,sans-serif;max-width:44rem;margin:12vh auto;
       padding:0 1.5rem;line-height:1.6;color:#1a1a19}
  code{background:#f0f0ec;padding:.15rem .4rem;border-radius:4px;font-size:.9em}
  pre{background:#f5f5f2;padding:1rem;border-radius:8px;overflow-x:auto}
</style></head><body>
<h1>El frontend no esta compilado</h1>
<p>Falta <code>static/dist</code>. El panel es una aplicacion React que hay que
construir antes de servirla:</p>
<pre>cd DASHBOARD_VOTACION/frontend
npm install
npm run build</pre>
<p>O bien levanta todo con Docker, que ya lo compila por ti:</p>
<pre>docker compose up --build</pre>
<p>La API sigue disponible: <a href="/api/estadisticas">/api/estadisticas</a></p>
</body></html>"""


def _servir_spa():
    indice = DIST / "index.html"
    if not indice.exists():
        return MENSAJE_SIN_BUILD, 503
    return send_from_directory(DIST, "index.html")


for _ruta in RUTAS_SPA:
    app.add_url_rule(_ruta, f"spa{_ruta.replace('/', '_')}", _servir_spa)


# --------------------------------------------------------------------------- #
#  API
# --------------------------------------------------------------------------- #
@app.route("/api/estadisticas")
def api_estadisticas():
    with _lock:
        return jsonify(dict(estadisticas_globales))


@app.route("/api/nodos")
def api_nodos():
    with _lock:
        return jsonify(dict(rendimiento_nodos))


@app.route("/api/serie")
def api_serie():
    """Evolución del recuento desde que arrancó el servidor (o el ciclo actual)."""
    with _lock:
        return jsonify(list(historial))


@app.route("/api/info")
def api_info():
    """Metadatos del modo de ejecucion, para que el frontend avise al visitante.

    `velocidad` importa: en replay el tiempo corre acelerado, asi que el panel
    tiene que dividir por ella para no anunciar un ritmo de votos inflado.
    """
    return jsonify({
        "modo": MODO,
        "grabando": GRABAR,
        "velocidad": VELOCIDAD_REPLAY if MODO == "replay" else 1.0,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "modo": MODO}), 200


@app.route("/api/update_stats", methods=["POST"])
def update_stats():
    datos = request.get_json(silent=True) or {}
    grabar_evento("stats", datos)
    aplicar_estadisticas(datos)
    return jsonify({"status": "success"}), 200


@app.route("/api/update_node", methods=["POST"])
def update_node():
    datos = request.get_json(silent=True) or {}
    grabar_evento("nodo", datos)
    aplicar_nodo(datos)
    return jsonify({"status": "success"}), 200


@app.route("/api/reset_stats", methods=["POST"])
def reset_stats():
    global _inicio_serie
    with _lock:
        estadisticas_globales.clear()
        estadisticas_globales.update(estado_inicial())
        rendimiento_nodos.clear()
        historial.clear()
        _inicio_serie = time.time()
        instantanea = dict(estadisticas_globales)
    socketio.emit("estadisticas_update", instantanea)
    socketio.emit("nodos_update", {})
    return jsonify({"status": "success"}), 200


# --------------------------------------------------------------------------- #
#  WebSocket
# --------------------------------------------------------------------------- #
@socketio.on("connect")
def on_connect():
    with _lock:
        stats = dict(estadisticas_globales)
        nodos = dict(rendimiento_nodos)
    emit("estadisticas_update", stats)
    emit("nodos_update", nodos)


# --------------------------------------------------------------------------- #
#  Arranque
# --------------------------------------------------------------------------- #
def iniciar_tareas_de_fondo():
    """Lanza el replay si toca. Se ejecuta tanto en local como bajo gunicorn."""
    if MODO == "replay":
        socketio.start_background_task(bucle_replay)


iniciar_tareas_de_fondo()


if __name__ == "__main__":
    destino = str(FICHERO_DEMO) if GRABAR else "no"
    print(" * Modo         : " + MODO)
    print(" * Concurrencia : " + ASYNC_MODE)
    print(" * Grabando     : " + destino)
    print(" * Escuchando en: http://localhost:" + str(PUERTO))
    socketio.run(app, host="0.0.0.0", port=PUERTO, debug=DEBUG, allow_unsafe_werkzeug=True)
