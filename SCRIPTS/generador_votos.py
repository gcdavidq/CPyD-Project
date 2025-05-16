# simulador_electoral_peru.py
"""
Generador de datos simulados para elecciones en Perú, incluyendo anomalías.
NODO DE SIMULACIÓN DE VOTACIÓN (COSTA y SIERRA NORTE)= 1
"""
import random
import csv
from datetime import datetime, timedelta
import numpy as np
import os
from typing import List, Dict, Set, Tuple, Any

class ConfiguracionElectoral:
    """Clase para manejar la configuración de la simulación electoral"""
    
    def __init__(self):
        # Regiones incluidas en la simulación
        self.regiones = [
            "Piura",
            "La Libertad",
            "Lambayeque", 
            "Cajamarca",
            "Amazonas",
            "Tumbes"
        ]
        
        # Candidatos/partidos participantes
        self.candidatos = ["APP", "APRA", "FUERZA POPULAR", "PERU LIBRE", "AVANZA PAIS"]
        
        # Población aproximada por región para ponderación
        self.poblacion_region = {
            "Piura": 1900000,
            "La Libertad": 1800000,
            "Lambayeque": 1200000,  
            "Cajamarca": 1400000,
            "Amazonas": 400000,
            "Tumbes": 200000
        }
        
        # Fecha y horarios de la elección
        self.fecha_eleccion = datetime(2025, 4, 6)
        self.hora_inicio = datetime(2025, 4, 6, 8, 0)  # 8:00 AM
        self.hora_fin = datetime(2025, 4, 6, 16, 0)    # 4:00 PM
        self.minutos_totales = int((self.hora_fin - self.hora_inicio).total_seconds() / 60)
        
        # Parámetros para DNIs peruanos
        self.min_dni = 10000000
        self.max_dni = 79999999
        
        # Parámetros para anomalías
        self.prob_minuto_anomalo = 0.03
        self.prob_region_anomala = 0.02
        self.tipos_anomalia = ["flujo_excesivo", "concentracion_candidato", "duplicados"]
        
        # Cálculo del flujo de votantes por región
        self.config_flujo = self._calcular_flujo_votantes()
        
        # Preferencias regionales (simula tendencias políticas por región)
        self.preferencias_regionales = self._generar_preferencias_regionales()
    
    def _calcular_flujo_votantes(self) -> Dict[str, Tuple[int, int]]:
        """Calcula el flujo promedio de votantes por minuto para cada región"""
        return {
            region: (int(poblacion * 0.00002), int(poblacion * 0.000005))
            for region, poblacion in self.poblacion_region.items()
        }
    
    def _generar_preferencias_regionales(self) -> Dict[str, List[float]]:
        """Genera preferencias aleatorias normalizadas por región para cada candidato"""
        preferencias = {
            region: [random.uniform(0.1, 0.3) for _ in range(len(self.candidatos))]
            for region in self.regiones
        }
        
        # Normalizar preferencias para que sumen 1
        for region in self.regiones:
            total = sum(preferencias[region])
            preferencias[region] = [p/total for p in preferencias[region]]
        
        return preferencias


class SimuladorElectoral:
    """Clase para simular el proceso electoral"""
    
    def __init__(self, config: ConfiguracionElectoral):
        self.config = config
        self.datos_votacion = []
    
    def generar_padron_electoral(self) -> Dict[str, List[int]]:
        """Genera el padrón electoral por región"""
        padron_electoral = {}
        dnis_usados = set()
        
        for region in self.config.regiones:
            # Asignar DNIs proporcionalmente a la población
            n_electores = int(self.config.poblacion_region[region] * 0.007)
            
            # Generar DNIs únicos para esta región
            dnis_region = set()
            while len(dnis_region) < n_electores:
                dni = random.randint(self.config.min_dni, self.config.max_dni)
                if dni not in dnis_usados:
                    dnis_region.add(dni)
                    dnis_usados.add(dni)
            
            padron_electoral[region] = list(dnis_region)
        
        return padron_electoral
    
    def simular_votacion(self) -> List[List[Any]]:
        """Ejecuta la simulación completa del proceso de votación"""
        # Predefinir regiones con anomalías regionales
        regiones_anomalas = {
            region: random.random() < self.config.prob_region_anomala 
            for region in self.config.regiones
        }
        
        # Generar padrón electoral
        padron_electoral = self.generar_padron_electoral()
        
        # Lista para almacenar todos los votos
        datos = []
        
        # Simular el día de votación por región
        for region in self.config.regiones:
            datos.extend(self._simular_votacion_region(
                region, 
                padron_electoral[region], 
                regiones_anomalas[region]
            ))
        
        # Ordenar por timestamp
        datos.sort(key=lambda x: x[0])
        return datos
    
    def _simular_votacion_region(self, region: str, padron_region: List[int], es_region_anomala: bool) -> List[List[Any]]:
        """Simula la votación en una región específica"""
        datos_region = []
        mu, sigma = self.config.config_flujo[region]
        preferencias_base = self.config.preferencias_regionales[region]
        
        dnis_disponibles = set(padron_region)
        dnis_votaron = set()
        
        # Simular voto por minuto
        for minuto in range(self.config.minutos_totales):
            timestamp = self.config.hora_inicio + timedelta(minutes=minuto)
            tiempo_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Determinar si este minuto tendrá anomalía
            es_minuto_anomalo = random.random() < self.config.prob_minuto_anomalo
            
            # Determinar tipo de anomalía (si hay)
            tipo_anomalia = None
            if es_minuto_anomalo or es_region_anomala:
                tipo_anomalia = random.choice(self.config.tipos_anomalia)
            
            datos_minuto = self._simular_minuto_votacion(
                tiempo_str, 
                region, 
                mu, 
                sigma, 
                preferencias_base,
                dnis_disponibles, 
                dnis_votaron, 
                minuto, 
                tipo_anomalia, 
                es_region_anomala
            )
            
            datos_region.extend(datos_minuto)
        
        return datos_region
    
    def _simular_minuto_votacion(
        self, 
        tiempo_str: str, 
        region: str, 
        mu: float, 
        sigma: float, 
        preferencias_base: List[float],
        dnis_disponibles: Set[int], 
        dnis_votaron: Set[int], 
        minuto: int, 
        tipo_anomalia: str, 
        es_region_anomala: bool
    ) -> List[List[Any]]:
        """Simula los votos emitidos en un minuto específico"""
        datos_minuto = []
        
        # Determinar cantidad de votos en este minuto
        if tipo_anomalia == "flujo_excesivo":
            # Anomalía: Muchos votos en poco tiempo
            votos_min = int(mu * random.uniform(3, 7))
        else:
            # Flujo normal con variación
            factor_hora = 1 + 0.3 * np.sin(np.pi * minuto / self.config.minutos_totales)  # Más votos a medio día
            votos_min = max(1, int(np.random.normal(mu * factor_hora, sigma)))
        
        # Determinar distribución de preferencias para este minuto
        probs = self._calcular_distribucion_preferencias(tipo_anomalia, preferencias_base)
        
        # Generar votos
        dnis_minuto_actual = set()
        for _ in range(votos_min):
            es_anomalo = tipo_anomalia is not None
            
            # Seleccionar DNI
            dni = self._seleccionar_dni(tipo_anomalia, dnis_votaron, dnis_disponibles)
            
            # Evitar duplicados en el mismo minuto (excepto si es anomalía)
            if dni in dnis_minuto_actual and tipo_anomalia != "duplicados":
                continue
            dnis_minuto_actual.add(dni)
            
            # Seleccionar candidato
            candidato = random.choices(self.config.candidatos, weights=probs)[0]
            
            # Registrar voto
            datos_minuto.append([tiempo_str, region, dni, candidato, int(es_anomalo)])
            
            # Para anomalías regionales, añadir más lentamente los DNIs usados
            if es_region_anomala and random.random() < 0.3:
                dni_falso = random.randint(self.config.min_dni, self.config.max_dni)
                datos_minuto.append([tiempo_str, region, dni_falso, candidato, 1])
        
        return datos_minuto
    
    def _calcular_distribucion_preferencias(self, tipo_anomalia: str, preferencias_base: List[float]) -> List[float]:
        """Calcula la distribución de preferencias según el tipo de anomalía"""
        if tipo_anomalia == "concentracion_candidato":
            # Anomalía: Concentración en un candidato
            candidato_favorecido = random.randint(0, len(self.config.candidatos) - 1)
            probs = [0.05] * len(self.config.candidatos)
            probs[candidato_favorecido] = 0.8
            sum_probs = sum(probs)
            return [p/sum_probs for p in probs]
        else:
            # Distribución normal con variación
            return preferencias_base
    
    def _seleccionar_dni(self, tipo_anomalia: str, dnis_votaron: Set[int], dnis_disponibles: Set[int]) -> int:
        """Selecciona un DNI según el tipo de anomalía"""
        if tipo_anomalia == "duplicados" and random.random() < 0.7 and dnis_votaron:
            # Anomalía: Votos duplicados (mismo DNI)
            return random.choice(list(dnis_votaron))
        elif dnis_disponibles:
            # Voto legítimo: DNI del padrón que no ha votado
            dni = random.choice(list(dnis_disponibles))
            dnis_disponibles.remove(dni)
            dnis_votaron.add(dni)
            return dni
        else:
            # Si se agotaron los DNIs legítimos, usar uno al azar
            return random.randint(self.config.min_dni, self.config.max_dni)
    
    def guardar_resultados(self, datos: List[List[Any]], archivo_salida: str = "votos_peru_simulados.csv") -> None:
        """Guarda los resultados de la simulación en un archivo CSV"""
        try:
            with open(archivo_salida, "w", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "region", "dni", "candidato", "anomalo"])
                writer.writerows(datos)
            print(f"Datos guardados exitosamente en {archivo_salida}")
        except IOError as e:
            print(f"Error al guardar el archivo: {e}")
    
    def generar_estadisticas(self, datos: List[List[Any]]) -> Dict[str, Any]:
        """Genera estadísticas sobre los resultados de la votación"""
        total_votos = len(datos)
        votos_anomalos = sum(1 for v in datos if v[4] == 1)
        porcentaje_anomalias = (votos_anomalos / total_votos) * 100
        
        # Conteo por candidato
        conteo_candidatos = {candidato: 0 for candidato in self.config.candidatos}
        for voto in datos:
            conteo_candidatos[voto[3]] += 1
        
        # Conteo por región
        conteo_regiones = {region: 0 for region in self.config.regiones}
        for voto in datos:
            conteo_regiones[voto[1]] += 1
        
        return {
            "total_votos": total_votos,
            "votos_anomalos": votos_anomalos,
            "porcentaje_anomalias": porcentaje_anomalias,
            "conteo_candidatos": conteo_candidatos,
            "conteo_regiones": conteo_regiones
        }
    
    def mostrar_estadisticas(self, estadisticas: Dict[str, Any]) -> None:
        """Muestra las estadísticas de manera formateada"""
        print("\nEstadísticas de la simulación:")
        print(f"Total de votos: {estadisticas['total_votos']}")
        print(f"Votos anómalos: {estadisticas['votos_anomalos']} ({estadisticas['porcentaje_anomalias']:.2f}%)")
        
        print("\nResultados por candidato:")
        for candidato, votos in estadisticas['conteo_candidatos'].items():
            porcentaje = (votos / estadisticas['total_votos']) * 100
            print(f"{candidato}: {votos} votos ({porcentaje:.2f}%)")
        
        print("\nResultados por región:")
        for region, votos in estadisticas['conteo_regiones'].items():
            porcentaje = (votos / estadisticas['total_votos']) * 100
            print(f"{region}: {votos} votos ({porcentaje:.2f}%)")


def main():
    """Función principal que ejecuta la simulación electoral"""
    # Crear configuración
    config = ConfiguracionElectoral()
    
    print(f"Iniciando simulación de elecciones peruanas para {len(config.regiones)} regiones y {len(config.candidatos)} candidatos")
    print(f"Período de votación: {config.hora_inicio.strftime('%H:%M')} a {config.hora_fin.strftime('%H:%M')} ({config.minutos_totales} minutos)")
    
    # Crear simulador
    simulador = SimuladorElectoral(config)
    
    # Generar datos de votación
    print("Generando datos de votación...")
    datos_votacion = simulador.simular_votacion()
    
    # Guardar resultados
    archivo_salida = "votos_peru_simulados.csv"
    print(f"Escribiendo {len(datos_votacion)} registros al archivo {archivo_salida}...")
    simulador.guardar_resultados(datos_votacion, archivo_salida)
    
    # Generar y mostrar estadísticas
    estadisticas = simulador.generar_estadisticas(datos_votacion)
    simulador.mostrar_estadisticas(estadisticas)
    
    print("\nSimulación completada.")


if __name__ == "__main__":
    main()