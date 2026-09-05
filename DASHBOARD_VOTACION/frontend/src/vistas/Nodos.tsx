import { useMemo } from "react";
import { TarjetaNodo } from "../componentes/TarjetaNodo";
import { Metrica, Tarjeta } from "../componentes/Ui";
import { decimal, numero, porcentaje } from "../datos/calculos";
import type { Datos } from "../datos/useDatos";

export function Nodos({ nodos, stats }: Datos) {
  const lista = useMemo(
    () => Object.values(nodos).sort((a, b) => a.nodo_id - b.nodo_id),
    [nodos],
  );

  const esclavos = lista.filter((n) => n.nodo_id !== 0);
  const maxLotes = Math.max(...esclavos.map((n) => n.lotes_completados), 1);
  const totalLotes = esclavos.reduce((a, n) => a + n.lotes_completados, 0);

  const cargaMedia = esclavos.length
    ? esclavos.reduce((a, n) => a + n.carga_actual, 0) / esclavos.length
    : 0;

  const hilosTotales = esclavos.reduce((a, n) => a + (n.numero_hilos || 0), 0);

  /**
   * Desequilibrio del reparto: cuánto se aleja el nodo más cargado de la media.
   * Es la métrica que justifica que exista el balanceador — si fuera 0, no
   * haría falta redistribuir nada.
   */
  const desequilibrio = useMemo(() => {
    if (esclavos.length < 2 || totalLotes === 0) return 0;
    const media = totalLotes / esclavos.length;
    const max = Math.max(...esclavos.map((n) => n.lotes_completados));
    return media > 0 ? (max - media) / media : 0;
  }, [esclavos, totalLotes]);

  if (lista.length === 0) {
    return (
      <Tarjeta titulo="Clúster">
        <p className="vacio">
          Ningún nodo ha reportado todavía. Lanza el clúster con
          <code className="mono"> mpirun -np 5 ./build/votacion</code>.
        </p>
      </Tarjeta>
    );
  }

  return (
    <>
      <div className="rejilla cuatro" style={{ marginBottom: 16 }}>
        <Metrica
          etiqueta="Nodos activos"
          valor={`${esclavos.length} + 1`}
          pie="esclavos + maestro"
        />
        <Metrica
          etiqueta="Hilos en paralelo"
          valor={numero(hilosTotales)}
          pie="sumando todos los esclavos"
        />
        <Metrica
          etiqueta="Carga media"
          valor={`${decimal(cargaMedia, 1)} %`}
          pie="uso real de CPU (getrusage)"
        />
        <Metrica
          etiqueta="Desequilibrio"
          valor={porcentaje(desequilibrio, 1)}
          pie="del nodo más cargado sobre la media"
        />
      </div>

      <h2
        style={{
          fontSize: 12,
          textTransform: "uppercase",
          letterSpacing: "0.055em",
          color: "var(--texto-3)",
          margin: "0 0 10px",
        }}
      >
        Nodos del clúster
      </h2>

      <div className="rejilla cuatro" style={{ marginBottom: 18 }}>
        {lista.map((n) => (
          <TarjetaNodo key={n.nodo_id} nodo={n} maxLotes={maxLotes} />
        ))}
      </div>

      <Tarjeta
        titulo="Detalle por nodo"
        nota="los mismos datos, en tabla"
        sinRelleno
      >
        <div className="scroll-x">
          <table className="tabla">
            <thead>
              <tr>
                <th>Nodo</th>
                <th>Rol</th>
                <th className="num">Lotes</th>
                <th className="num">Cuota</th>
                <th className="num">Tiempo/lote</th>
                <th className="num">Carga CPU</th>
                <th className="num">Hilos</th>
                <th>GPU</th>
              </tr>
            </thead>
            <tbody>
              {lista.map((n) => (
                <tr key={n.nodo_id}>
                  <td className="mono">rank {n.nodo_id}</td>
                  <td>{n.nodo_id === 0 ? "Maestro" : "Esclavo"}</td>
                  <td className="num">{n.nodo_id === 0 ? "—" : numero(n.lotes_completados)}</td>
                  <td className="num">
                    {n.nodo_id === 0 || totalLotes === 0
                      ? "—"
                      : porcentaje(n.lotes_completados / totalLotes, 1)}
                  </td>
                  <td className="num">
                    {n.tiempo_promedio_lote > 0 ? `${decimal(n.tiempo_promedio_lote, 3)} s` : "—"}
                  </td>
                  <td className="num">{decimal(n.carga_actual, 1)} %</td>
                  <td className="num">{n.numero_hilos || "—"}</td>
                  <td>{n.tiene_gpu ? "CUDA" : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Tarjeta>

      <p style={{ marginTop: 16, fontSize: 12, color: "var(--texto-3)", maxWidth: 780 }}>
        El maestro consulta estas cifras cada 15 segundos. Si encuentra un nodo por encima del
        50 % de carga y otro por debajo del 30 %, le pide al saturado que ceda un cuarto de su
        cola y se la reenvía al que está libre. Las marcas sobre cada barra de CPU son
        precisamente esos dos umbrales. Total procesado hasta ahora:{" "}
        <strong className="cifra">{numero(stats.total_votos)}</strong> votos.
      </p>
    </>
  );
}
