# simulador_electoral_peru_optimizado.py
"""
Generador de datos simulados para elecciones en Perú, incluyendo anomalías.
NODO DE SIMULACIÓN DE VOTACIÓN (COSTA y SIERRA NORTE)= 1
VERSIÓN OPTIMIZADA
"""
import random
import csv
from datetime import datetime, timedelta
import numpy as np
import os
from typing import List, Dict, Set, Tuple, Any
#DATOS DE POBLACION QUE PUEDE VOTAR: 70%
#QUIENES REALMENTE VOTAN: 70% de ese 70%
class ConfiguracionElectoral:
    """Clase para manejar la configuración de la simulación electoral"""
    
    def __init__(self):
        # Regiones incluidas en la simulación
        self.regiones = [
            "Piura",
            "Lambayeque",
            "La Libertad",
            "San Martín",
            "Ucayali",
            "Tumbes",
            "Amazonas"
        ]
        # Candidatos/partidos participantes
        self.candidatos = ["APP", "APRA", "FUERZA POPULAR", "PERU LIBRE", "AVANZA PAIS"]
        
        # Población aproximada por región para ponderación
        self.poblacion_region = {
            "Piura": 2160800,
            "Lambayeque": 1400700,
            "La Libertad": 2073200,
            "San Martín": 939400,
            "Ucayali": 571200,
            "Tumbes": 263000,
            "Amazonas": 433200
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
        
        # Pre-calcular timestamps para mejor rendimiento
        self.timestamps = self._precalcular_timestamps()
    
    def _calcular_flujo_votantes(self) -> Dict[str, Tuple[int, int]]:
        """Calcula el flujo promedio de votantes por minuto para cada región"""
        return {
            region: (
                int(poblacion * 0.7 * 0.9 / self.minutos_totales), 
                int(poblacion * 0.7 * 0.9 / self.minutos_totales * 0.3)
            )
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
    
    def _precalcular_timestamps(self) -> List[str]:
        """Pre-calcula todos los timestamps para evitar recalcular en cada iteración"""
        timestamps = []
        for minuto in range(self.minutos_totales):
            timestamp = self.hora_inicio + timedelta(minutes=minuto)
            timestamps.append(timestamp.strftime("%Y-%m-%d %H:%M:%S"))
        return timestamps


class SimuladorElectoral:
    """Clase para simular el proceso electoral"""
    
    def __init__(self, config: ConfiguracionElectoral):
        self.config = config
        self.datos_votacion = []
        # Pre-calcular factores de hora para mejor rendimiento
        self.factores_hora = self._precalcular_factores_hora()
    
    def _precalcular_factores_hora(self) -> List[float]:
        """Pre-calcula los factores de hora para evitar cálculos repetitivos"""
        return [1 + 0.3 * np.sin(np.pi * minuto / self.config.minutos_totales) 
                for minuto in range(self.config.minutos_totales)]
    
    def generar_padron_electoral(self) -> Dict[str, np.ndarray]:
        """Genera el padrón electoral por región usando arrays de numpy para mejor rendimiento"""
        padron_electoral = {}
        
        # Generar todos los DNIs de una vez
        total_electores = sum(int(self.config.poblacion_region[region] * 0.8) 
                            for region in self.config.regiones)
        
        # Generar rango de DNIs únicos
        dnis_pool = np.random.choice(
            range(self.config.min_dni, self.config.max_dni + 1),
            size=total_electores * 2,  # Generar más para asegurar unicidad
            replace=False
        )
        
        # Distribuir DNIs por región
        dni_idx = 0
        for region in self.config.regiones:
            n_electores = int(self.config.poblacion_region[region] * 0.8)
            padron_electoral[region] = dnis_pool[dni_idx:dni_idx + n_electores]
            dni_idx += n_electores
        
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
        
        # Lista para almacenar todos los votos - pre-asignar capacidad estimada
        datos = []
        capacidad_estimada = sum(
            int(self.config.poblacion_region[region] * 0.8 * 0.8)  # ~80% participación estimada
            for region in self.config.regiones
        )
        
        # Simular el día de votación por región
        for region in self.config.regiones:
            datos.extend(self._simular_votacion_region(
                region, 
                padron_electoral[region], 
                regiones_anomalas[region]
            ))
        
        # Ordenar por timestamp usando key más eficiente
        datos.sort(key=lambda x: x[0])
        return datos
    
    def _simular_votacion_region(self, region: str, padron_region: np.ndarray, es_region_anomala: bool) -> List[List[Any]]:
        """Simula la votación en una región específica"""
        datos_region = []
        mu, sigma = self.config.config_flujo[region]
        preferencias_base = self.config.preferencias_regionales[region]
        
        # Convertir a lista solo una vez y usar índices para mejor rendimiento
        padron_lista = padron_region.tolist()
        dnis_disponibles_idx = list(range(len(padron_lista)))
        random.shuffle(dnis_disponibles_idx)  # Mezclar una sola vez
        
        dnis_votaron = []
        dni_idx_usado = 0
        
        # Pre-generar todas las decisiones de anomalías para evitar llamadas repetitivas a random
        anomalias_minuto = [random.random() < self.config.prob_minuto_anomalo 
                          for _ in range(self.config.minutos_totales)]
        tipos_anomalia_minuto = [random.choice(self.config.tipos_anomalia) 
                               for _ in range(self.config.minutos_totales)]
        
        # Simular voto por minuto
        for minuto in range(self.config.minutos_totales):
            tiempo_str = self.config.timestamps[minuto]
            
            # Determinar si este minuto tendrá anomalía
            es_minuto_anomalo = anomalias_minuto[minuto]
            
            # Determinar tipo de anomalía (si hay)
            tipo_anomalia = None
            if es_minuto_anomalo or es_region_anomala:
                tipo_anomalia = tipos_anomalia_minuto[minuto]
            
            datos_minuto, dni_idx_usado = self._simular_minuto_votacion_optimizado(
                tiempo_str, 
                region, 
                mu, 
                sigma, 
                preferencias_base,
                padron_lista,
                dnis_disponibles_idx,
                dni_idx_usado,
                dnis_votaron, 
                minuto, 
                tipo_anomalia, 
                es_region_anomala
            )
            
            datos_region.extend(datos_minuto)
        
        return datos_region
    
    def _simular_minuto_votacion_optimizado(
        self, 
        tiempo_str: str, 
        region: str, 
        mu: float, 
        sigma: float, 
        preferencias_base: List[float],
        padron_lista: List[int],
        dnis_disponibles_idx: List[int],
        dni_idx_usado: int,
        dnis_votaron: List[int], 
        minuto: int, 
        tipo_anomalia: str, 
        es_region_anomala: bool
    ) -> Tuple[List[List[Any]], int]:
        """Simula los votos emitidos en un minuto específico - versión optimizada"""
        datos_minuto = []
        
        # Determinar cantidad de votos en este minuto
        if tipo_anomalia == "flujo_excesivo":
            # Anomalía: Muchos votos en poco tiempo
            votos_min = int(mu * random.uniform(3, 7))
        else:
            # Usar factor pre-calculado
            factor_hora = self.factores_hora[minuto]
            votos_min = max(1, int(np.random.normal(mu * factor_hora, sigma)))
        
        # Determinar distribución de preferencias para este minuto
        probs = self._calcular_distribucion_preferencias(tipo_anomalia, preferencias_base)
        
        # Pre-generar decisiones aleatorias para mejor rendimiento
        candidatos_elegidos = np.random.choice(
            self.config.candidatos, 
            size=votos_min, 
            p=probs
        )
        
        # Generar votos
        dnis_minuto_actual = set()
        es_anomalo = 1 if tipo_anomalia is not None else 0
        
        for i in range(votos_min):
            # Seleccionar DNI
            dni, dni_idx_usado = self._seleccionar_dni_optimizado(
                tipo_anomalia, 
                dnis_votaron, 
                padron_lista,
                dnis_disponibles_idx,
                dni_idx_usado
            )
            
            # Evitar duplicados en el mismo minuto (excepto si es anomalía)
            if dni in dnis_minuto_actual and tipo_anomalia != "duplicados":
                continue
            dnis_minuto_actual.add(dni)
            
            # Usar candidato pre-generado
            candidato = candidatos_elegidos[i]
            
            # Registrar voto
            datos_minuto.append([tiempo_str, region, dni, candidato, es_anomalo])
            
            # Para anomalías regionales, añadir más lentamente los DNIs usados
            if es_region_anomala and random.random() < 0.3:
                dni_falso = random.randint(self.config.min_dni, self.config.max_dni)
                datos_minuto.append([tiempo_str, region, dni_falso, candidato, 1])
        
        return datos_minuto, dni_idx_usado
    
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
    
    def _seleccionar_dni_optimizado(
        self, 
        tipo_anomalia: str, 
        dnis_votaron: List[int], 
        padron_lista: List[int],
        dnis_disponibles_idx: List[int],
        dni_idx_usado: int
    ) -> Tuple[int, int]:
        """Selecciona un DNI según el tipo de anomalía - versión optimizada"""
        if tipo_anomalia == "duplicados" and random.random() < 0.7 and dnis_votaron:
            # Anomalía: Votos duplicados (mismo DNI)
            return random.choice(dnis_votaron), dni_idx_usado
        elif dni_idx_usado < len(dnis_disponibles_idx):
            # Voto legítimo: DNI del padrón que no ha votado
            idx = dnis_disponibles_idx[dni_idx_usado]
            dni = padron_lista[idx]
            dnis_votaron.append(dni)
            return dni, dni_idx_usado + 1
        else:
            # Si se agotaron los DNIs legítimos, usar uno al azar
            return random.randint(self.config.min_dni, self.config.max_dni), dni_idx_usado
    
    def guardar_resultados(self, datos: List[List[Any]], archivo_salida: str = "votos_peru_simulados.csv") -> None:
        """Guarda los resultados de la simulación en un archivo CSV con buffer optimizado"""
        try:
            with open(archivo_salida, "w", newline='', encoding='utf-8', buffering=8192) as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "region", "dni", "candidato", "anomalo"])
                
                # Escribir en lotes para mejor rendimiento
                batch_size = 1000
                for i in range(0, len(datos), batch_size):
                    batch = datos[i:i + batch_size]
                    writer.writerows(batch)
                    
            print(f"Datos guardados exitosamente en {archivo_salida}")
        except IOError as e:
            print(f"Error al guardar el archivo: {e}")
    
    def generar_estadisticas(self, datos: List[List[Any]]) -> Dict[str, Any]:
        """Genera estadísticas sobre los resultados de la votación - versión optimizada"""
        total_votos = len(datos)
        
        # Usar contadores más eficientes
        votos_anomalos = sum(1 for v in datos if v[4] == 1)
        porcentaje_anomalias = (votos_anomalos / total_votos) * 100 if total_votos > 0 else 0
        
        # Conteo por candidato usando dict comprehension
        conteo_candidatos = {candidato: 0 for candidato in self.config.candidatos}
        conteo_regiones = {region: 0 for region in self.config.regiones}
        
        # Un solo bucle para ambos conteos
        for voto in datos:
            conteo_candidatos[voto[3]] += 1
            conteo_regiones[voto[1]] += 1
        
        # Calcular estadísticas adicionales
        votos_por_minuto_region = self._calcular_media_votacion_por_minuto(datos)
        votacion_por_region_candidato = self._calcular_votacion_por_region_candidato(datos)
        
        return {
            "total_votos": total_votos,
            "votos_anomalos": votos_anomalos,
            "porcentaje_anomalias": porcentaje_anomalias,
            "conteo_candidatos": conteo_candidatos,
            "conteo_regiones": conteo_regiones,
            "media_votos_por_minuto_region": votos_por_minuto_region,
            "votacion_por_region_candidato": votacion_por_region_candidato
        }
    
    def _calcular_media_votacion_por_minuto(self, datos: List[List[Any]]) -> Dict[str, float]:
        """Calcula la media de votación por minuto para cada región"""
        votos_por_minuto_region = {}
        
        # Inicializar contadores
        for region in self.config.regiones:
            votos_por_minuto_region[region] = {}
        
        # Contar votos por minuto y región
        for voto in datos:
            timestamp = voto[0]
            region = voto[1]
            
            if timestamp not in votos_por_minuto_region[region]:
                votos_por_minuto_region[region][timestamp] = 0
            votos_por_minuto_region[region][timestamp] += 1
        
        # Calcular promedios
        medias = {}
        for region in self.config.regiones:
            if votos_por_minuto_region[region]:
                total_votos = sum(votos_por_minuto_region[region].values())
                total_minutos = len(votos_por_minuto_region[region])
                medias[region] = total_votos / total_minutos if total_minutos > 0 else 0
            else:
                medias[region] = 0
        
        return medias
    
    def _calcular_votacion_por_region_candidato(self, datos: List[List[Any]]) -> Dict[str, Dict[str, int]]:
        """Calcula la votación por región y candidato dentro de cada región"""
        votacion_region_candidato = {}
        
        # Inicializar estructura
        for region in self.config.regiones:
            votacion_region_candidato[region] = {candidato: 0 for candidato in self.config.candidatos}
        
        # Contar votos
        for voto in datos:
            region = voto[1]
            candidato = voto[3]
            votacion_region_candidato[region][candidato] += 1
        
        return votacion_region_candidato
    
    def mostrar_estadisticas(self, estadisticas: Dict[str, Any]) -> None:
        """Muestra las estadísticas de manera formateada"""
        print("\nEstadísticas de la simulación:")
        print(f"Total de votos: {estadisticas['total_votos']}")
        print(f"Votos anómalos: {estadisticas['votos_anomalos']} ({estadisticas['porcentaje_anomalias']:.2f}%)")
        
        print("\nResultados por candidato:")
        for candidato, votos in estadisticas['conteo_candidatos'].items():
            porcentaje = (votos / estadisticas['total_votos']) * 100 if estadisticas['total_votos'] > 0 else 0
            print(f"{candidato}: {votos} votos ({porcentaje:.2f}%)")
        
        print("\nResultados por región:")
        for region, votos in estadisticas['conteo_regiones'].items():
            porcentaje = (votos / estadisticas['total_votos']) * 100 if estadisticas['total_votos'] > 0 else 0
            print(f"{region}: {votos} votos ({porcentaje:.2f}%)")
        
        # ESTADÍSTICAS ADICIONALES SOLICITADAS
        print("\n" + "="*60)
        print("ESTADÍSTICAS ADICIONALES")
        print("="*60)
        
        # Media de votación por minuto por región
        print("\nMedia de votación por minuto por región:")
        print("-" * 45)
        for region, media in estadisticas['media_votos_por_minuto_region'].items():
            print(f"{region}: {media:.2f} votos/minuto")
        
        # Votación por región y candidatos dentro de ellas
        print("\nVotación por candidatos dentro de cada región:")
        print("-" * 52)
        for region, candidatos_votos in estadisticas['votacion_por_region_candidato'].items():
            print(f"\n{region.upper()}:")
            total_region = sum(candidatos_votos.values())
            for candidato, votos in candidatos_votos.items():
                porcentaje_region = (votos / total_region) * 100 if total_region > 0 else 0
                print(f"  {candidato}: {votos} votos ({porcentaje_region:.2f}% de la región)")


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
    archivo_salida = "votos_simulados100_region1.csv"
    print(f"Escribiendo {len(datos_votacion)} registros al archivo {archivo_salida}...")
    simulador.guardar_resultados(datos_votacion, archivo_salida)
    
    # Generar y mostrar estadísticas
    estadisticas = simulador.generar_estadisticas(datos_votacion)
    simulador.mostrar_estadisticas(estadisticas)
    
    print("\nSimulación completada.")


if __name__ == "__main__":
    main()