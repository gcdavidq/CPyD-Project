import { useState } from "react";
import { numero, porcentaje, type Entrada } from "../datos/calculos";
import "./BarrasRanking.css";

interface Props {
  datos: Entrada[];
  /** Cuántas filas se muestran antes de agrupar el resto en "otras". */
  limite?: number;
  /** Etiqueta de la fila agregada. */
  etiquetaResto?: string;
  /** Se llama al pulsar una fila (para filtrar el resto del panel). */
  onSeleccion?: (clave: string | null) => void;
  seleccion?: string | null;
  vacio?: string;
}

/**
 * Ranking con barra de proporción.
 *
 * Una sola serie, así que un solo color: la categoría ya está en la etiqueta
 * de cada fila. Pintar cada barra de un color distinto codificaría la posición,
 * que no es un dato.
 */
export function BarrasRanking({
  datos,
  limite = 8,
  etiquetaResto = "Resto",
  onSeleccion,
  seleccion = null,
  vacio = "Sin datos todavía",
}: Props) {
  const [encima, setEncima] = useState<string | null>(null);

  if (datos.length === 0) {
    return <p className="vacio">{vacio}</p>;
  }

  const total = datos.reduce((a, d) => a + d.valor, 0);
  const visibles = datos.slice(0, limite);
  const resto = datos.slice(limite);
  const sumaResto = resto.reduce((a, d) => a + d.valor, 0);
  const maximo = Math.max(...visibles.map((d) => d.valor), 1);

  const filas: (Entrada & { agregada?: boolean })[] = [...visibles];
  if (sumaResto > 0) {
    filas.push({ clave: `${etiquetaResto} (${resto.length})`, valor: sumaResto, agregada: true });
  }

  return (
    <ul className="ranking">
      {filas.map((d) => {
        const proporcion = d.valor / maximo;
        const activa = seleccion !== null && seleccion === d.clave;
        const pulsable = Boolean(onSeleccion) && !d.agregada;

        return (
          <li
            key={d.clave}
            className={`ranking-fila${activa ? " activa" : ""}${d.agregada ? " agregada" : ""}${pulsable ? " pulsable" : ""}`}
            onMouseEnter={() => setEncima(d.clave)}
            onMouseLeave={() => setEncima(null)}
            onClick={pulsable ? () => onSeleccion?.(activa ? null : d.clave) : undefined}
            role={pulsable ? "button" : undefined}
            tabIndex={pulsable ? 0 : undefined}
            onKeyDown={
              pulsable
                ? (e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      onSeleccion?.(activa ? null : d.clave);
                    }
                  }
                : undefined
            }
            title={`${d.clave}: ${numero(d.valor)} votos${total > 0 ? ` · ${porcentaje(d.valor / total)}` : ""}`}
          >
            <span className="ranking-nombre">{d.clave}</span>
            <span className="ranking-pista">
              <span
                className="ranking-barra"
                style={{ width: `${Math.max(proporcion * 100, 1.5)}%` }}
              />
            </span>
            <span className="ranking-valor cifra">{numero(d.valor)}</span>
            <span className="ranking-cuota cifra">
              {total > 0 ? porcentaje(d.valor / total, 1) : "—"}
            </span>
            {encima === d.clave && <span className="solo-lectores">seleccionado</span>}
          </li>
        );
      })}
    </ul>
  );
}
