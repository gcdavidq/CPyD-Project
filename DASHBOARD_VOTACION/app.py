from flask import Flask, render_template, jsonify, send_from_directory, request
from flask_socketio import SocketIO, emit
import threading, time, random
from datetime import datetime
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'votacion_electronica_peru'
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Globals & Manager ---
estadisticas_globales = {
    'total_votos': 0,
    'anomalias_reales': 0,
    'anomalias_detectadas': 0,
    'falsos_positivos': 0,
    'falsos_negativos': 0,
    'votos_por_region': {},
    'votos_por_candidato': {},
    'votos_por_candidato_por_region': {},
    'anomalias_por_region_candidato': {},
    'anomalias_por_candidato_region': {},
    'nodos_activos': 0,
    'ultimo_update': datetime.now().strftime('%H:%M:%S'),
    'tiempo_procesamiento': 0
}

# Pre-pobla algunas regiones para la demo
CANDIDATOS = ['Candidato A', 'Candidato B', 'Candidato C']
REGIONES = ['Lima', 'Cusco', 'Arequipa', 'Puno']

# Inicializa votos por región y por candidato
for region in REGIONES:
    estadisticas_globales['votos_por_region'][region] = 0
    estadisticas_globales['votos_por_candidato_por_region'][region] = {}
    for candidato in CANDIDATOS:
        estadisticas_globales['votos_por_candidato_por_region'][region][candidato] = 0
        estadisticas_globales['votos_por_candidato'].setdefault(candidato, 0)

class EstadisticasManager:
    def __init__(self):
        self.lock = threading.Lock()
    def actualizar_estadisticas(self, nuevas):
        with self.lock:
            estadisticas_globales.update(nuevas)
            estadisticas_globales['ultimo_update'] = datetime.now().strftime('%H:%M:%S')
            socketio.emit('estadisticas_update', estadisticas_globales)
    def agregar_votos_region(self, region, qty):
        with self.lock:
            estadisticas_globales['votos_por_region'].setdefault(region,0)
            estadisticas_globales['votos_por_region'][region] += qty
            estadisticas_globales['total_votos'] += qty
            estadisticas_globales['ultimo_update'] = datetime.now().strftime('%H:%M:%S')
            socketio.emit('estadisticas_update', estadisticas_globales)

stats_manager = EstadisticasManager()
rendimiento_nodos = {}

POPULARIDAD = {
    'Lima':       [0.6, 0.3, 0.1],  # Candidato A más popular
    'Cusco':      [0.2, 0.6, 0.2],  # Candidato B más popular
    'Arequipa':   [0.3, 0.3, 0.4],  # Candidato C más popular
    'Puno':       [0.4, 0.2, 0.4],  # A y C igualados
}

PESOS_REGION = [0.5, 0.2, 0.2, 0.1] 

def simular_datos():
    while True:
        time.sleep(2)
        region = random.choices(REGIONES, weights=PESOS_REGION)[0]
        popularidad = POPULARIDAD[region]
        candidato = random.choices(CANDIDATOS, weights=popularidad)[0]
        cantidad = random.randint(1, 10)
        with stats_manager.lock:
            estadisticas_globales['votos_por_region'][region] += cantidad
            estadisticas_globales['votos_por_candidato'][candidato] += cantidad
            estadisticas_globales['votos_por_candidato_por_region'][region][candidato] += cantidad
            estadisticas_globales['total_votos'] += cantidad
            estadisticas_globales['ultimo_update'] = datetime.now().strftime('%H:%M:%S')
            socketio.emit('estadisticas_update', estadisticas_globales)

#threading.Thread(target=simular_datos, daemon=True).start()

# --- Web Routes ---
@app.route('/')
def index():       return render_template('index.html')
@app.route('/mapa')
def mapa():         return render_template('contenido.html')
@app.route('/help')
def help_page():    return render_template('help.html')
@app.route('/login')
def login():        return render_template('iniciar_sesion.html')
@app.route('/register')
def register():     return render_template('registrarse.html')

# --- API / Socket ---
@app.route('/api/estadisticas')
def api_estadisticas(): return jsonify(estadisticas_globales)

@socketio.on('connect')
def on_connect():   emit('estadisticas_update', estadisticas_globales)

@app.route('/api/update_stats', methods=['POST'])
def update_stats():
    data = request.get_json()
    stats_manager.actualizar_estadisticas(data)
    return jsonify({'status':'success'}), 200

@app.route('/api/update_node', methods=['POST'])
def update_node():
    data = request.get_json()
    nodo_id = data.get('nodo_id')
    if nodo_id is not None:
        rendimiento_nodos[nodo_id] = data
        socketio.emit('nodos_update', rendimiento_nodos)
    return jsonify({'status':'success'}), 200

@app.route('/api/reset_stats', methods=['POST'])
def reset_stats():
    global estadisticas_globales, rendimiento_nodos
    estadisticas_globales = { k:0 for k in estadisticas_globales }
    rendimiento_nodos = {}
    socketio.emit('estadisticas_update', estadisticas_globales)
    socketio.emit('nodos_update', rendimiento_nodos)
    return jsonify({'status':'success'}), 200

if __name__ == '__main__':
    os.makedirs('static/data', exist_ok=True)
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
