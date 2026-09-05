import { useMemo, useRef, useState } from "react";
import { compacto, numero } from "../datos/calculos";
import type { Muestra } from "../tipos";
import "./AreaTemporal.css";

interface Props {
  serie: Muestra[];
  altura?: number;
}

const MARGEN = { arriba: 14, derecha: 10, abajo: 22, izquierda: 46 };

/**
 * Evolución del recuento durante la jornada.
 *
 * Una sola serie, así que no lleva leyenda: el título de la tarjeta ya dice
 * qué se está midiendo. Lleva crosshair y tooltip porque es un gráfico HTML y
 * el lector espera poder inspeccionar un punto concreto.
 */
export function AreaTemporal({ serie, altura = 190 }: Props) {
  const [indice, setIndice] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const ancho = 720; // viewBox fijo; el SVG escala al contenedor

  const geo = useMemo(() => {
    if (serie.length < 2) return null;

    const x0 = MARGEN.izquierda;
    const x1 = ancho - MARGEN.derecha;
    const y0 = altura - MARGEN.abajo;
    const y1 = MARGEN.arriba;

    const tMax = Math.max(serie[serie.length - 1].t, 1);
    const vMax = Math.max(...serie.map((m) => m.votos), 1);

    const px = (t: number) => x0 + (t / tMax) * (x1 - x0);
    const py = (v: number) => y0 - (v / vMax) * (y0 - y1);

    const linea = serie.map((m, i) => `${i === 0 ? "M" : "L"}${px(m.t).toFixed(1)} ${py(m.votos).toFixed(1)}`).join(" ");
    const area = `${linea} L${px(serie[serie.length - 1].t).toFixed(1)} ${y0} L${px(serie[0].t).toFixed(1)} ${y0} Z`;

    // Tres marcas en el eje Y: suficientes para dar escala sin cargar el fondo.
    const marcas = [0, vMax / 2, vMax].map((v) => ({ v, y: py(v) }));

    return { x0, x1, y0, y1, px, py, linea, area, marcas, tMax, vMax };
  }, [serie, altura]);

  if (!geo) {
    // Con una sola muestra todavía no hay línea que dibujar. El mensaje
    // distingue "no hay datos" de "aún no hay suficientes".
    return (
      <p className="vacio">
        {serie.length === 0
          ? "Esperando datos del clúster…"
          : "Acumulando muestras para la gráfica…"}
      </p>
    );
  }

  const alPasar = (e: React.MouseEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const caja = svg.getBoundingClientRect();
    const x = ((e.clientX - caja.left) / caja.width) * ancho;
    // Punto más cercano en X, que es como se lee una serie temporal.
    let mejor = 0;
    let dist = Infinity;
    serie.forEach((m, i) => {
      const d = Math.abs(geo.px(m.t) - x);
      if (d < dist) {
        dist = d;
        mejor = i;
      }
    });
    setIndice(mejor);
  };

  const activo = indice !== null ? serie[indice] : null;

  return (
    <div className="area-envoltorio">
      <svg
        ref={svgRef}
        viewBox={`0 0 ${ancho} ${altura}`}
        className="area-svg"
        preserveAspectRatio="none"
        onMouseMove={alPasar}
        onMouseLeave={() => setIndice(null)}
        role="img"
        aria-label={`Evolución del recuento: ${numero(serie[serie.length - 1].votos)} votos acumulados`}
      >
        <defs>
          <linearGradient id="degradadoArea" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--acento)" stopOpacity="0.22" />
            <stop offset="100%" stopColor="var(--acento)" stopOpacity="0.01" />
          </linearGradient>
        </defs>

        {geo.marcas.map((m, i) => (
          <g key={i}>
            <line
              x1={geo.x0}
              y1={m.y}
              x2={geo.x1}
              y2={m.y}
              className="area-rejilla"
              vectorEffect="non-scaling-stroke"
            />
            <text x={geo.x0 - 8} y={m.y + 4} className="area-etiqueta" textAnchor="end">
              {compacto(m.v)}
            </text>
          </g>
        ))}

        <path d={geo.area} fill="url(#degradadoArea)" />
        <path
          d={geo.linea}
          className="area-linea"
          fill="none"
          vectorEffect="non-scaling-stroke"
        />

        {activo && indice !== null && (
          <g>
            <line
              x1={geo.px(activo.t)}
              y1={geo.y1}
              x2={geo.px(activo.t)}
              y2={geo.y0}
              className="area-cruz"
              vectorEffect="non-scaling-stroke"
            />
            <circle
              cx={geo.px(activo.t)}
              cy={geo.py(activo.votos)}
              r={4}
              className="area-punto"
              vectorEffect="non-scaling-stroke"
            />
          </g>
        )}
      </svg>

      {activo && (
        <div
          className="area-tooltip"
          style={{
            left: `${(geo.px(activo.t) / ancho) * 100}%`,
          }}
        >
          <strong className="cifra">{numero(activo.votos)}</strong> votos
          <span className="area-tooltip-sec cifra">
            {numero(activo.anomalias)} anomalías · {(activo.t / 1000).toFixed(0)} s
          </span>
        </div>
      )}
    </div>
  );
}
