import { useEffect, useMemo, useRef, useState } from "react";
import { buscarClave, compacto, numero, porcentaje } from "../datos/calculos";
import "./MapaPeru.css";

type Anillo = [number, number][];

interface Departamento {
  nombre: string;
  anillos: Anillo[];
}

interface Props {
  /** Valor a representar por departamento (votos, anomalías…). */
  valores: Record<string, number>;
  seleccion: string | null;
  onSeleccion: (departamento: string | null) => void;
  /** Se muestra en el tooltip detrás de la cifra. */
  unidad?: string;
}

const ANCHO = 520;
const ALTO = 640;
const RAMPA = ["--ramp-0", "--ramp-1", "--ramp-2", "--ramp-3", "--ramp-4", "--ramp-5"];

/**
 * Coropleta del Perú dibujada directamente en SVG.
 *
 * Se prescinde de Leaflet a propósito: no hace falta un mapa navegable con
 * teselas, y una coropleta comunica mucho mejor "dónde hay más votos" que un
 * puñado de marcadores idénticos. Además evita una dependencia y las llamadas
 * de red a un servidor de teselas.
 *
 * La escala es secuencial (un único tono de claro a oscuro), que es la
 * correcta para una magnitud. Un arcoíris sugeriría categorías donde solo hay
 * "más" y "menos".
 */
export function MapaPeru({ valores, seleccion, onSeleccion, unidad = "votos" }: Props) {
  const [deps, setDeps] = useState<Departamento[]>([]);
  const [error, setError] = useState(false);
  const [encima, setEncima] = useState<string | null>(null);
  const [raton, setRaton] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const cajaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let vivo = true;
    fetch("/static/data/departamentos_per%C3%BA.geojson")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("no disponible"))))
      .then((geo) => {
        if (!vivo) return;
        const lista: Departamento[] = (geo.features ?? []).map((f: any) => {
          const g = f.geometry;
          const anillos: Anillo[] =
            g.type === "Polygon"
              ? g.coordinates
              : g.coordinates.flatMap((p: Anillo[]) => p);
          return { nombre: f.properties?.NOMBDEP ?? "?", anillos };
        });
        setDeps(lista);
      })
      .catch(() => vivo && setError(true));
    return () => {
      vivo = false;
    };
  }, []);

  const proyeccion = useMemo(() => {
    if (deps.length === 0) return null;

    let minLon = Infinity, maxLon = -Infinity, minLat = Infinity, maxLat = -Infinity;
    for (const d of deps) {
      for (const anillo of d.anillos) {
        for (const [lon, lat] of anillo) {
          if (lon < minLon) minLon = lon;
          if (lon > maxLon) maxLon = lon;
          if (lat < minLat) minLat = lat;
          if (lat > maxLat) maxLat = lat;
        }
      }
    }

    // Equirectangular con corrección por latitud: a la escala del Perú la
    // distorsión frente a una proyección propiamente dicha es imperceptible.
    const latMedia = ((minLat + maxLat) / 2) * (Math.PI / 180);
    const anchoGeo = (maxLon - minLon) * Math.cos(latMedia);
    const altoGeo = maxLat - minLat;

    const margen = 12;
    const escala = Math.min(
      (ANCHO - margen * 2) / anchoGeo,
      (ALTO - margen * 2) / altoGeo,
    );

    const desplX = (ANCHO - anchoGeo * escala) / 2;
    const desplY = (ALTO - altoGeo * escala) / 2;

    const px = (lon: number) => desplX + (lon - minLon) * Math.cos(latMedia) * escala;
    const py = (lat: number) => desplY + (maxLat - lat) * escala;

    const camino = (d: Departamento) =>
      d.anillos
        .map(
          (anillo) =>
            anillo
              .map(([lon, lat], i) => `${i === 0 ? "M" : "L"}${px(lon).toFixed(1)} ${py(lat).toFixed(1)}`)
              .join(" ") + " Z",
        )
        .join(" ");

    return { camino };
  }, [deps]);

  const maximo = useMemo(
    () => Math.max(...Object.values(valores).map((v) => Number(v) || 0), 1),
    [valores],
  );

  const total = useMemo(
    () => Object.values(valores).reduce((a, b) => a + (Number(b) || 0), 0),
    [valores],
  );

  const valorDe = (nombre: string): number | null => {
    const clave = buscarClave(valores, nombre);
    return clave === null ? null : Number(valores[clave]) || 0;
  };

  /** Escala por raíz: sin ella, Lima aplasta al resto y el mapa queda plano. */
  const tono = (v: number | null): string => {
    if (v === null) return "var(--superficie-alt)";
    if (v <= 0) return `var(${RAMPA[0]})`;
    const t = Math.sqrt(v / maximo);
    const i = Math.min(RAMPA.length - 1, Math.max(1, Math.round(t * (RAMPA.length - 1))));
    return `var(${RAMPA[i]})`;
  };

  if (error) {
    return <p className="vacio">No se pudo cargar el mapa de departamentos.</p>;
  }
  if (!proyeccion) {
    return <p className="vacio">Cargando mapa…</p>;
  }

  const activo = encima ?? seleccion;
  const valorActivo = activo ? valorDe(activo) : null;

  return (
    <div className="mapa" ref={cajaRef}>
      <svg
        viewBox={`0 0 ${ANCHO} ${ALTO}`}
        className="mapa-svg"
        role="img"
        aria-label="Mapa del Perú por departamento"
        onMouseLeave={() => setEncima(null)}
      >
        {deps.map((d) => {
          const v = valorDe(d.nombre);
          const seleccionado = seleccion !== null && buscarClave({ [seleccion]: 0 }, d.nombre) !== null;
          const resaltado = encima === d.nombre;
          return (
            <path
              key={d.nombre}
              d={proyeccion.camino(d)}
              fill={tono(v)}
              className={`mapa-dep${seleccionado ? " seleccionado" : ""}${resaltado ? " encima" : ""}${v === null ? " sin-datos" : ""}`}
              vectorEffect="non-scaling-stroke"
              onMouseEnter={() => setEncima(d.nombre)}
              onMouseMove={(e) => {
                const caja = cajaRef.current?.getBoundingClientRect();
                if (caja) setRaton({ x: e.clientX - caja.left, y: e.clientY - caja.top });
              }}
              onClick={() => v !== null && onSeleccion(seleccion && seleccionado ? null : d.nombre)}
              tabIndex={v === null ? -1 : 0}
              onKeyDown={(e) => {
                if ((e.key === "Enter" || e.key === " ") && v !== null) {
                  e.preventDefault();
                  onSeleccion(seleccion && seleccionado ? null : d.nombre);
                }
              }}
            >
              <title>
                {d.nombre}: {v === null ? "sin datos" : `${numero(v)} ${unidad}`}
              </title>
            </path>
          );
        })}
      </svg>

      {activo && (
        <div
          className="mapa-tooltip"
          style={{ left: raton.x, top: raton.y }}
          role="status"
        >
          <strong>{activo}</strong>
          <span className="cifra">
            {valorActivo === null
              ? "Sin datos"
              : `${numero(valorActivo)} ${unidad}${total > 0 ? ` · ${porcentaje(valorActivo / total, 1)}` : ""}`}
          </span>
        </div>
      )}

      <div className="mapa-leyenda">
        <span className="mapa-leyenda-texto">0</span>
        <span className="mapa-leyenda-rampa">
          {RAMPA.map((r) => (
            <i key={r} style={{ background: `var(${r})` }} />
          ))}
        </span>
        <span className="mapa-leyenda-texto">{compacto(maximo)}</span>
        <span className="mapa-leyenda-unidad">{unidad}</span>
      </div>
    </div>
  );
}
