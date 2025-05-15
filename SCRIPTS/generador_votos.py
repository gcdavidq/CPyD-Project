import random
import csv
from datetime import datetime, timedelta
import numpy as np

# Configuración
REGIONES = [
    "Lima", "Arequipa", "Cusco", "La Libertad", "Piura", 
    "Lambayeque", "Junín", "Cajamarca", "Puno", "Loreto"
]

CANDIDATOS = ["A", "B", "C", "D", "E"]

# Definir la población aproximada por región para ponderación
POBLACION_REGION = {
    "Lima": 10000000,
    "Arequipa": 1500000,
    "Cusco": 1300000,
    "La Libertad": 2000000,
    "Piura": 2000000,
    "Lambayeque": 1300000,
    "Junín": 1400000,
    "Cajamarca": 1500000,
    "Puno": 1400000,
    "Loreto": 1000000
}

# Configuración del flujo promedio de votantes por minuto por región
# (media, desviación estándar)
CONFIG_FLUJO = {
    region: (int(poblacion * 0.00002), int(poblacion * 0.000005)) 
    for region, poblacion in POBLACION_REGION.items()
}

# Fecha de la elección
FECHA_ELECCION = datetime(2025, 4, 6)
HORA_INICIO = datetime(2025, 4, 6, 8, 0)  # 8:00 AM
HORA_FIN = datetime(2025, 4, 6, 16, 0)    # 4:00 PM
MINUTOS_TOTALES = int((HORA_FIN - HORA_INICIO).total_seconds() / 60)

# Parámetros para anomalías
PROB_MINUTO_ANOMALO = 0.03
PROB_REGION_ANOMALA = 0.02
TIPOS_ANOMALIA = ["flujo_excesivo", "concentracion_candidato", "duplicados"]

# DNI válidos en Perú (8 dígitos)
MIN_DNI = 10000000
MAX_DNI = 79999999

# Preferencias regionales (simula tendencias políticas por región)
PREFERENCIAS_REGIONALES = {
    region: [random.uniform(0.1, 0.3) for _ in range(len(CANDIDATOS))]
    for region in REGIONES
}

# Normalizar preferencias para que sumen 1
for region in REGIONES:
    total = sum(PREFERENCIAS_REGIONALES[region])
    PREFERENCIAS_REGIONALES[region] = [p/total for p in PREFERENCIAS_REGIONALES[region]]

def generar_datos_votacion():
    """Genera los datos completos de la votación"""
    datos = []
    dnis_usados = set()  # Para evitar duplicados legítimos
    
    # Predefinir regiones con anomalías regionales
    regiones_anomalas = {region: random.random() < PROB_REGION_ANOMALA for region in REGIONES}
    
    # Distribuir DNIs válidos entre regiones (simula padrón electoral)
    padron_electoral = {}
    for region in REGIONES:
        # Asignar DNIs proporcionalmente a la población
        n_electores = int(POBLACION_REGION[region] * 0.007)  
        
        # Generar DNIs únicos para esta región
        dnis_region = set()
        while len(dnis_region) < n_electores:
            dni = random.randint(MIN_DNI, MAX_DNI)
            if dni not in dnis_usados:
                dnis_region.add(dni)
                dnis_usados.add(dni)
        
        padron_electoral[region] = list(dnis_region)
    
    # Simular el día de votación
    for region in REGIONES:
        mu, sigma = CONFIG_FLUJO[region]
        preferencias_base = PREFERENCIAS_REGIONALES[region]
        dnis_disponibles = set(padron_electoral[region])
        dnis_votaron = set()
        
        # Simular voto por minuto
        for minuto in range(MINUTOS_TOTALES):
            timestamp = HORA_INICIO + timedelta(minutes=minuto)
            tiempo_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Determinar si este minuto tendrá anomalía
            es_minuto_anomalo = random.random() < PROB_MINUTO_ANOMALO
            es_region_anomala = regiones_anomalas[region]
            
            # Determinar tipo de anomalía (si hay)
            tipo_anomalia = None
            if es_minuto_anomalo or es_region_anomala:
                tipo_anomalia = random.choice(TIPOS_ANOMALIA)
            
            # Determinar cantidad de votos en este minuto
            if tipo_anomalia == "flujo_excesivo":
                # Anomalía: Muchos votos en poco tiempo
                votos_min = int(mu * random.uniform(3, 7))
            else:
                # Flujo normal con variación
                factor_hora = 1 + 0.3 * np.sin(np.pi * minuto / MINUTOS_TOTALES)  # Más votos a medio día
                votos_min = max(1, int(np.random.normal(mu * factor_hora, sigma)))
            
            # Determinar distribución de preferencias para este minuto
            if tipo_anomalia == "concentracion_candidato":
                # Anomalía: Concentración en un candidato
                candidato_favorecido = random.randint(0, len(CANDIDATOS) - 1)
                probs = [0.05] * len(CANDIDATOS)
                probs[candidato_favorecido] = 0.8
                sum_probs = sum(probs)
                probs = [p/sum_probs for p in probs]
            else:
                # Distribución normal con variación
                probs = preferencias_base
            
            # Generar votos
            dnis_minuto_actual = set()
            for _ in range(votos_min):
                es_anomalo = tipo_anomalia is not None
                
                # Seleccionar DNI
                if tipo_anomalia == "duplicados" and random.random() < 0.7 and dnis_votaron:
                    # Anomalía: Votos duplicados (mismo DNI)
                    dni = random.choice(list(dnis_votaron))
                elif dnis_disponibles:
                    # Voto legítimo: DNI del padrón que no ha votado
                    dni = random.choice(list(dnis_disponibles))
                    dnis_disponibles.remove(dni)
                    dnis_votaron.add(dni)
                else:
                    # Si se agotaron los DNIs legítimos, usar uno al azar
                    dni = random.randint(MIN_DNI, MAX_DNI)
                
                # Evitar duplicados en el mismo minuto (excepto si es anomalía)
                if dni in dnis_minuto_actual and tipo_anomalia != "duplicados":
                    continue
                dnis_minuto_actual.add(dni)
                
                # Seleccionar candidato
                candidato = random.choices(CANDIDATOS, weights=probs)[0]
                
                # Registrar voto
                datos.append([tiempo_str, region, dni, candidato, int(es_anomalo)])
                
                # Para anomalías regionales, añadir más lentamente los DNIs usados
                if es_region_anomala and random.random() < 0.3:
                    dni_falso = random.randint(MIN_DNI, MAX_DNI)
                    datos.append([tiempo_str, region, dni_falso, candidato, 1])
    
    return datos

def main():
    print(f"Iniciando simulación de elecciones peruanas para {len(REGIONES)} regiones y {len(CANDIDATOS)} candidatos")
    print(f"Período de votación: {HORA_INICIO.strftime('%H:%M')} a {HORA_FIN.strftime('%H:%M')} ({MINUTOS_TOTALES} minutos)")
    
    # Generar datos
    print("Generando datos de votación...")
    datos_votacion = generar_datos_votacion()
    
    # Ordenar por timestamp
    datos_votacion.sort(key=lambda x: x[0])
    
    # Guardar resultados
    archivo_salida = "votos_peru_simulados.csv"
    print(f"Escribiendo {len(datos_votacion)} registros al archivo {archivo_salida}...")
    
    with open(archivo_salida, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "region", "dni", "candidato", "anomalo"])
        writer.writerows(datos_votacion)
    
    # Estadísticas básicas
    total_votos = len(datos_votacion)
    votos_anomalos = sum(1 for v in datos_votacion if v[4] == 1)
    porcentaje_anomalias = (votos_anomalos / total_votos) * 100
    
    # Conteo por candidato
    conteo_candidatos = {candidato: 0 for candidato in CANDIDATOS}
    for voto in datos_votacion:
        conteo_candidatos[voto[3]] += 1
    
    print("\nEstadísticas de la simulación:")
    print(f"Total de votos: {total_votos}")
    print(f"Votos anómalos: {votos_anomalos} ({porcentaje_anomalias:.2f}%)")
    print("\nResultados por candidato:")
    for candidato, votos in conteo_candidatos.items():
        porcentaje = (votos / total_votos) * 100
        print(f"{candidato}: {votos} votos ({porcentaje:.2f}%)")
    
    print("\nSimulación completada.")

if __name__ == "__main__":
    main()