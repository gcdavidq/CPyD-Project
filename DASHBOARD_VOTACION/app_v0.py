from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import json
import threading
import time
import random
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'votacion_electronica_peru'
socketio = SocketIO(app, cors_allowed_origins="*")

# Datos globales de estadísticas
estadisticas_globales = {
    'total_votos': 0,
    'anomalias_reales': 0,
    'anomalias_detectadas': 0,
    'falsos_positivos': 0,
    'falsos_negativos': 0,
    'votos_por_region': {},
    'votos_por_candidato': {},
    'votos_por_candidato_por_region': {},  # Nuevo
    'anomalias_por_region_candidato': {},  # Nuevo
    'anomalias_por_candidato_region': {},  # Nuevo
    'nodos_activos': 0,
    'ultimo_update': datetime.now().strftime('%H:%M:%S'),
    'tiempo_procesamiento': 0
}

# Datos de rendimiento por nodo
rendimiento_nodos = {}

# Regiones del Perú
REGIONES_PERU = [
    'Lima', 'Arequipa', 'La Libertad', 'Piura', 'Lambayeque', 'Cusco',
    'Junín', 'Cajamarca', 'Ancash', 'Loreto', 'Huánuco', 'San Martín',
    'Ica', 'Ayacucho', 'Ucayali', 'Puno', 'Tacna', 'Tumbes',
    'Apurímac', 'Huancavelica', 'Moquegua', 'Pasco', 'Amazonas',
    'Madre de Dios', 'Callao'
]

# Candidatos ficticios
CANDIDATOS = [ "APP", "APRA", "FUERZA POPULAR", "PERU LIBRE", "AVANZA PAIS"
]

class EstadisticasManager:
    def __init__(self):
        self.lock = threading.Lock()
    
    def actualizar_estadisticas(self, nuevas_stats):
        """Actualiza las estadísticas globales"""
        with self.lock:
            global estadisticas_globales
            estadisticas_globales.update(nuevas_stats)
            estadisticas_globales['ultimo_update'] = datetime.now().strftime('%H:%M:%S')
            
            # Emitir actualización a todos los clientes conectados
            socketio.emit('estadisticas_update', estadisticas_globales)
    
    def agregar_votos_region(self, region, cantidad):
        """Agrega votos a una región específica"""
        with self.lock:
            if region not in estadisticas_globales['votos_por_region']:
                estadisticas_globales['votos_por_region'][region] = 0
            estadisticas_globales['votos_por_region'][region] += cantidad
    
    def agregar_votos_candidato(self, candidato, cantidad):
        """Agrega votos a un candidato específico"""
        with self.lock:
            if candidato not in estadisticas_globales['votos_por_candidato']:
                estadisticas_globales['votos_por_candidato'][candidato] = 0
            estadisticas_globales['votos_por_candidato'][candidato] += cantidad

# Inicializar manager
stats_manager = EstadisticasManager()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/estadisticas')
def get_estadisticas():
    """API endpoint para obtener estadísticas actuales"""
    return jsonify(estadisticas_globales)

@app.route('/api/nodos')
def get_nodos():
    """API endpoint para obtener información de nodos"""
    return jsonify(rendimiento_nodos)

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')
    emit('estadisticas_update', estadisticas_globales)

@socketio.on('disconnect')
def handle_disconnect():
    print('Cliente desconectado')

def simular_datos():
    """Simula la recepción de datos de tu aplicación C++"""
    while True:
        time.sleep(2)  # Actualizar cada 2 segundos
        
        # Simular nuevos votos
        nuevos_votos = random.randint(50, 200)
        region_aleatoria = random.choice(REGIONES_PERU)
        candidato_aleatorio = random.choice(CANDIDATOS)
        
        # Actualizar estadísticas
        stats_manager.agregar_votos_region(region_aleatoria, nuevos_votos)
        stats_manager.agregar_votos_candidato(candidato_aleatorio, nuevos_votos)
        
        # Simular anomalías ocasionales
        if random.random() < 0.1:  # 10% de probabilidad
            anomalias = random.randint(1, 5)
            estadisticas_globales['anomalias_reales'] += anomalias
            estadisticas_globales['anomalias_detectadas'] += anomalias - random.randint(0, 1)
        
        estadisticas_globales['total_votos'] += nuevos_votos
        estadisticas_globales['nodos_activos'] = random.randint(3, 8)
        estadisticas_globales['tiempo_procesamiento'] += random.uniform(0.1, 0.5)
        
        # Calcular métricas
        if estadisticas_globales['anomalias_reales'] > 0:
            estadisticas_globales['falsos_positivos'] = max(0, 
                estadisticas_globales['anomalias_detectadas'] - estadisticas_globales['anomalias_reales'])
            estadisticas_globales['falsos_negativos'] = max(0, 
                estadisticas_globales['anomalias_reales'] - estadisticas_globales['anomalias_detectadas'])
        
        stats_manager.actualizar_estadisticas(estadisticas_globales)

# Agregar estos endpoints a tu archivo Flask existente

@app.route('/api/update_stats', methods=['POST'])
def update_stats():
    """Endpoint para recibir estadísticas desde C++"""
    try:
        data = request.get_json()
        print("→ Datos recibidos:", data.keys())  # [Opcional] Depuración
        
        # Actualizar estadísticas globales
        global estadisticas_globales
        estadisticas_globales.update(data)
        estadisticas_globales['ultimo_update'] = datetime.now().strftime('%H:%M:%S')
        
        # Emitir actualización a todos los clientes conectados
        socketio.emit('estadisticas_update', estadisticas_globales)
        
        return jsonify({'status': 'success', 'message': 'Estadísticas actualizadas'}), 200
        
    except Exception as e:
        print(f"Error actualizando estadísticas: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/update_node', methods=['POST'])
def update_node():
    """Endpoint para recibir información de rendimiento de nodos"""
    try:
        data = request.get_json()
        nodo_id = data.get('nodo_id')
        
        if nodo_id is not None:
            rendimiento_nodos[nodo_id] = data
            
            # Emitir actualización de nodos
            socketio.emit('nodos_update', rendimiento_nodos)
            
        return jsonify({'status': 'success'}), 200
        
    except Exception as e:
        print(f"Error actualizando nodo: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/reset_stats', methods=['POST'])
def reset_stats():
    """Endpoint para reiniciar estadísticas"""
    global estadisticas_globales, rendimiento_nodos
    
    estadisticas_globales = {
        'total_votos': 0,
        'anomalias_reales': 0,
        'anomalias_detectadas': 0,
        'falsos_positivos': 0,
        'falsos_negativos': 0,
        'votos_por_region': {},
        'votos_por_candidato': {},
        'votos_por_candidato_por_region': {},  # Nuevo
        'anomalias_por_region_candidato': {},  # Nuevo
        'anomalias_por_candidato_region': {},  # Nuevo
        'nodos_activos': 0,
        'ultimo_update': datetime.now().strftime('%H:%M:%S'),
        'tiempo_procesamiento': 0
    }
    
    rendimiento_nodos = {}
    
    # Emitir reset a todos los clientes
    socketio.emit('estadisticas_update', estadisticas_globales)
    socketio.emit('nodos_update', rendimiento_nodos)
    
    return jsonify({'status': 'success', 'message': 'Estadísticas reiniciadas'}), 200

# Función para recibir datos desde tu aplicación C++
def recibir_datos_cpp(stats_data):
    """
    Esta función debe ser llamada desde tu aplicación C++ 
    para actualizar las estadísticas reales
    """
    stats_manager.actualizar_estadisticas(stats_data)

if __name__ == '__main__':
    
    # Inicializar algunas regiones y candidatos
    for region in REGIONES_PERU[:10]:
        estadisticas_globales['votos_por_region'][region] = random.randint(100, 1000)
    
    for candidato in CANDIDATOS[:6]:
        estadisticas_globales['votos_por_candidato'][candidato] = random.randint(500, 5000)
    
    # Iniciar simulación en hilo separado (eliminar en producción)
    threading.Thread(target=simular_datos, daemon=True).start()

    
    # Crear directorio templates si no existe
    os.makedirs('templates', exist_ok=True)
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
