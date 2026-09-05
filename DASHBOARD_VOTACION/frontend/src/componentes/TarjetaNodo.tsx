import { decimal, estadoCarga, numero } from "../datos/calculos";
import type { Nodo } from "../tipos";
import "./TarjetaNodo.css";

interface Props {
  nodo: Nodo;
  /** Lotes del nodo más cargado, para escalar la barra de reparto. */
  maxLotes: number;
}

const ETIQUETA_CARGA: Record<string, string> = {
  bien: "Holgado",
  aviso: "Cargado",
  grave: "Saturado",
};

/**
 * Ficha de un nodo del clúster.
 *
 * Los umbrales de color son los mismos que usa el balanceador en C++
 * (>50 % cargado, >80 % saturado), de modo que lo que se ve en pantalla
 * coincide con la decisión que el maestro está tomando.
 */
export function TarjetaNodo({ nodo, maxLotes }: Props) {
  const esMaestro = nodo.nodo_id === 0;
  const estado = estadoCarga(nodo.carga_actual);
  const carga = Math.min(100, Math.max(0, nodo.carga_actual));

  return (
    <article className={`nodo${esMaestro ? " nodo-maestro" : ""}`}>
      <header className="nodo-cabecera">
        <span className="nodo-nombre">
          {esMaestro ? "Maestro" : `Esclavo ${nodo.nodo_id}`}
          <span className="nodo-rank mono">rank {nodo.nodo_id}</span>
        </span>
        <span className={`pill ${esMaestro ? "neutro" : estado}`}>
          {esMaestro ? "Coordina" : ETIQUETA_CARGA[estado]}
        </span>
      </header>

      <div className="nodo-carga">
        <div className="nodo-carga-fila">
          <span className="nodo-carga-etiqueta">Uso de CPU</span>
          <span className="nodo-carga-valor cifra">{decimal(carga, 1)} %</span>
        </div>
        <div className="nodo-pista" role="img" aria-label={`Uso de CPU ${decimal(carga, 1)} por ciento`}>
          <span className={`nodo-relleno ${estado}`} style={{ width: `${Math.max(carga, 1)}%` }} />
          {/* Marcas de los umbrales que dispara el balanceador */}
          <span className="nodo-umbral" style={{ left: "50%" }} />
          <span className="nodo-umbral" style={{ left: "80%" }} />
        </div>
      </div>

      <dl className="nodo-datos">
        <div>
          <dt>Lotes</dt>
          <dd className="cifra">{numero(nodo.lotes_completados)}</dd>
        </div>
        <div>
          <dt>s / lote</dt>
          <dd className="cifra">
            {nodo.tiempo_promedio_lote > 0 ? decimal(nodo.tiempo_promedio_lote, 2) : "—"}
          </dd>
        </div>
        <div>
          <dt>Hilos</dt>
          <dd className="cifra">{nodo.numero_hilos || "—"}</dd>
        </div>
        <div>
          <dt>GPU</dt>
          <dd>{nodo.tiene_gpu ? <span className="pill bien">CUDA</span> : <span className="nodo-no">No</span>}</dd>
        </div>
      </dl>

      {!esMaestro && (
        <div className="nodo-reparto">
          <span className="nodo-reparto-etiqueta">Reparto de lotes</span>
          <span className="nodo-reparto-pista">
            <span
              className="nodo-reparto-barra"
              style={{ width: `${maxLotes > 0 ? Math.max((nodo.lotes_completados / maxLotes) * 100, 1.5) : 0}%` }}
            />
          </span>
        </div>
      )}
    </article>
  );
}
