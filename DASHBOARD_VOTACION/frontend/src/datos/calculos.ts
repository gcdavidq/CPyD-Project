import type { Confusion, Estadisticas, MapaAnidado } from "../tipos";

/* ------------------------------------------------------------------ formato */

const nf = new Intl.NumberFormat("es-PE");

export const numero = (n: number | undefined | null): string =>
  nf.format(Math.round(n ?? 0));

export const porcentaje = (n: number, decimales = 1): string =>
  `${(n * 100).toFixed(decimales).replace(".", ",")} %`;

export const decimal = (n: number, decimales = 2): string =>
  n.toFixed(decimales).replace(".", ",");

/** Abrevia cifras grandes para los ejes: 1 200 000 -> "1,2 M". */
export function compacto(n: number): string {
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1).replace(".", ",")} M`;
  if (Math.abs(n) >= 1_000) return `${Math.round(n / 1_000)} k`;
  return String(Math.round(n));
}

export function duracion(segundos: number): string {
  if (!segundos || segundos < 1) return "—";
  const m = Math.floor(segundos / 60);
  const s = Math.round(segundos % 60);
  return m > 0 ? `${m} min ${s} s` : `${s} s`;
}

/* ---------------------------------------------------- matriz de confusión -- */

/**
 * Reconstruye las cuatro celdas a partir de los contadores que envía el
 * backend.
 *
 *   VP = anomalías reales que sí se detectaron
 *   VN = votos normales que no se marcaron
 *
 * Las divisiones van protegidas: con el clúster recién arrancado todos los
 * denominadores son cero.
 */
export function confusion(s: Estadisticas): Confusion {
  const vp = Math.max(0, s.anomalias_reales - s.falsos_negativos);
  const fp = s.falsos_positivos;
  const fn = s.falsos_negativos;
  const vn = Math.max(0, s.total_votos - s.anomalias_reales - fp);

  const div = (a: number, b: number) => (b > 0 ? a / b : 0);

  const precision = div(vp, vp + fp);
  const recall = div(vp, vp + fn);
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  const accuracy = div(vp + vn, s.total_votos);

  return { vp, fp, fn, vn, precision, recall, f1, accuracy };
}

/* ----------------------------------------------------------- ordenaciones -- */

export interface Entrada {
  clave: string;
  valor: number;
}

/** Convierte un diccionario en una lista ordenada de mayor a menor. */
export function ordenar(mapa: Record<string, number> | undefined): Entrada[] {
  if (!mapa) return [];
  return Object.entries(mapa)
    .map(([clave, valor]) => ({ clave, valor: Number(valor) || 0 }))
    .sort((a, b) => b.valor - a.valor);
}

/** Aplana un mapa anidado sumando el segundo nivel. */
export function totalizar(mapa: MapaAnidado | undefined): Record<string, number> {
  const salida: Record<string, number> = {};
  for (const [externo, interno] of Object.entries(mapa ?? {})) {
    salida[externo] = Object.values(interno ?? {}).reduce((a, b) => a + (Number(b) || 0), 0);
  }
  return salida;
}

/* ------------------------------------------------------------ nombres ----- */

/**
 * Los nombres de departamento llegan de dos sitios con formatos distintos: el
 * GeoJSON los trae en mayúsculas y sin tildes ("LA LIBERTAD"), y el clúster
 * capitalizados ("La Libertad"). Se normalizan ambos lados antes de comparar.
 */
export function normalizar(nombre: string | undefined | null): string {
  if (!nombre) return "";
  return nombre
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toUpperCase()
    .replace(/\s+/g, " ")
    .trim();
}

/** Busca en un diccionario la clave equivalente a `nombre`, ignorando tildes. */
export function buscarClave<T>(
  mapa: Record<string, T> | undefined,
  nombre: string,
): string | null {
  if (!mapa) return null;
  const objetivo = normalizar(nombre);
  return Object.keys(mapa).find((k) => normalizar(k) === objetivo) ?? null;
}

/* ------------------------------------------------------------- derivados -- */

/** Nivel de carga de un nodo, usando los mismos umbrales que el balanceador. */
export function estadoCarga(carga: number): "bien" | "aviso" | "grave" {
  if (carga >= 80) return "grave";
  if (carga >= 50) return "aviso";
  return "bien";
}

/** Votos por segundo entre las dos últimas muestras de la serie. */
export function ritmo(serie: { t: number; votos: number }[]): number {
  if (serie.length < 2) return 0;
  const a = serie[serie.length - 2];
  const b = serie[serie.length - 1];
  const dt = (b.t - a.t) / 1000;
  return dt > 0 ? Math.max(0, (b.votos - a.votos) / dt) : 0;
}
