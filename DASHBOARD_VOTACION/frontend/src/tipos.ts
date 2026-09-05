/** Formas de los datos que publica el backend Flask. */

/** Mapa anidado: primer nivel región (o candidato), segundo el contrario. */
export type MapaAnidado = Record<string, Record<string, number>>;

/** Lo que devuelve `GET /api/estadisticas` y emite el evento `estadisticas_update`. */
export interface Estadisticas {
  total_votos: number;
  anomalias_reales: number;
  anomalias_detectadas: number;
  falsos_positivos: number;
  falsos_negativos: number;
  votos_por_region: Record<string, number>;
  votos_por_candidato: Record<string, number>;
  votos_por_candidato_por_region: MapaAnidado;
  anomalias_por_region_candidato: MapaAnidado;
  anomalias_por_candidato_region: MapaAnidado;
  nodos_activos: number;
  ultimo_update: string;
  tiempo_procesamiento: number;
}

/** Un nodo del clúster, tal como lo reporta el maestro por HTTP. */
export interface Nodo {
  nodo_id: number;
  tiempo_promedio_lote: number;
  lotes_completados: number;
  tiene_gpu: boolean;
  carga_actual: number;
  numero_hilos: number;
}

/** `GET /api/nodos` devuelve un diccionario indexado por id de nodo. */
export type Nodos = Record<string, Nodo>;

/** `GET /api/info`: en qué modo está funcionando el backend. */
export interface Info {
  modo: "live" | "replay" | string;
  grabando: boolean;
  /** Factor de aceleracion del replay; 1.0 cuando hay un cluster real detras. */
  velocidad?: number;
}

/** Un punto de la serie temporal que el panel va acumulando en el cliente. */
export interface Muestra {
  t: number;        // milisegundos desde que se abrió la página
  votos: number;
  anomalias: number;
}

/**
 * Matriz de confusión derivada de los contadores acumulados.
 *
 * El backend envía anomalías reales/detectadas y los dos tipos de error; de
 * ahí se reconstruyen las cuatro celdas y las métricas asociadas.
 */
export interface Confusion {
  vp: number;
  fp: number;
  fn: number;
  vn: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
}

export const ESTADISTICAS_VACIAS: Estadisticas = {
  total_votos: 0,
  anomalias_reales: 0,
  anomalias_detectadas: 0,
  falsos_positivos: 0,
  falsos_negativos: 0,
  votos_por_region: {},
  votos_por_candidato: {},
  votos_por_candidato_por_region: {},
  anomalias_por_region_candidato: {},
  anomalias_por_candidato_region: {},
  nodos_activos: 0,
  ultimo_update: "--:--:--",
  tiempo_procesamiento: 0,
};
