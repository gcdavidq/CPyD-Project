import { useMemo, useState } from "react";
import { AreaTemporal } from "../componentes/AreaTemporal";
import { BarrasRanking } from "../componentes/BarrasRanking";
import { MatrizConfusion } from "../componentes/MatrizConfusion";
import { Metrica, Tarjeta } from "../componentes/Ui";
import {
  buscarClave,
  confusion,
  decimal,
  numero,
  ordenar,
  porcentaje,
  ritmo,
  totalizar,
} from "../datos/calculos";
import type { Datos } from "../datos/useDatos";

export function Resumen({ stats, nodos, serie, info }: Datos) {
  const [region, setRegion] = useState<string | null>(null);

  const c = useMemo(() => confusion(stats), [stats]);
  const porRegion = useMemo(() => ordenar(stats.votos_por_region), [stats]);
  const anomaliasPorRegion = useMemo(
    () => ordenar(totalizar(stats.anomalias_por_region_candidato)),
    [stats],
  );

  // Cuando hay un departamento seleccionado, el reparto por candidato pasa a
  // ser el de esa región; si no, el global.
  const porCandidato = useMemo(() => {
    if (region) {
      const clave = buscarClave(stats.votos_por_candidato_por_region, region);
      if (clave) return ordenar(stats.votos_por_candidato_por_region[clave]);
    }
    return ordenar(stats.votos_por_candidato);
  }, [stats, region]);

  const lotes = useMemo(
    () => Object.values(nodos).reduce((a, n) => a + (n.lotes_completados || 0), 0),
    [nodos],
  );

  // En modo replay el reloj corre acelerado, asi que el ritmo medido hay que
  // dividirlo por el factor de reproduccion para que sea el real.
  const velocidad = ritmo(serie) / Math.max(info?.velocidad ?? 1, 0.01);
  const esclavos = Object.values(nodos).filter((n) => n.nodo_id !== 0).length;
  const tasaAnomalias = stats.total_votos > 0 ? stats.anomalias_reales / stats.total_votos : 0;

  return (
    <>
      <div className="rejilla cuatro" style={{ marginBottom: 16 }}>
        <Metrica
          etiqueta="Votos procesados"
          valor={numero(stats.total_votos)}
          pie={velocidad > 0 ? `${numero(velocidad)} votos/s` : "En espera"}
        />
        <Metrica
          etiqueta="Lotes completados"
          valor={numero(lotes)}
          pie={`${esclavos} nodo${esclavos === 1 ? "" : "s"} procesando`}
        />
        <Metrica
          etiqueta="Anomalías detectadas"
          valor={numero(stats.anomalias_detectadas)}
          pie={`${numero(stats.anomalias_reales)} reales · ${porcentaje(tasaAnomalias, 1)} del padrón`}
        />
        <Metrica
          etiqueta="Precisión / Recall"
          valor={
            <>
              {porcentaje(c.precision, 1)}
              <span style={{ color: "var(--texto-3)", fontWeight: 400 }}> / </span>
              {porcentaje(c.recall, 1)}
            </>
          }
          pie={`F1 ${porcentaje(c.f1, 1)}`}
        />
      </div>

      <div className="rejilla ancha-estrecha" style={{ marginBottom: 16 }}>
        <Tarjeta
          titulo="Evolución del recuento"
          nota={serie.length > 1 ? `${serie.length} actualizaciones` : "Acumulando…"}
        >
          <AreaTemporal serie={serie} />
        </Tarjeta>

        <Tarjeta
          titulo="Calidad de la detección"
          nota={<span title="Sobre el total de votos procesados">matriz de confusión</span>}
        >
          <MatrizConfusion c={c} />
        </Tarjeta>
      </div>

      <div className="rejilla tres">
        <Tarjeta
          titulo="Votos por departamento"
          nota={`${porRegion.length} con datos`}
        >
          <BarrasRanking
            datos={porRegion}
            limite={8}
            etiquetaResto="Otros departamentos"
            seleccion={region}
            onSeleccion={setRegion}
          />
        </Tarjeta>

        <Tarjeta
          titulo={region ? `Candidatos en ${region}` : "Votos por candidato"}
          nota={
            region ? (
              <button
                onClick={() => setRegion(null)}
                style={{
                  border: "none",
                  background: "none",
                  color: "var(--acento)",
                  padding: 0,
                  fontSize: 12,
                }}
              >
                ver global
              </button>
            ) : (
              "nacional"
            )
          }
        >
          <BarrasRanking datos={porCandidato} limite={6} />
        </Tarjeta>

        <Tarjeta
          titulo="Anomalías por departamento"
          nota="votos marcados"
        >
          <BarrasRanking
            datos={anomaliasPorRegion}
            limite={8}
            etiquetaResto="Otros departamentos"
            vacio="Aún no se han detectado anomalías"
          />
        </Tarjeta>
      </div>

      <p
        style={{
          marginTop: 18,
          fontSize: 12,
          color: "var(--texto-3)",
          maxWidth: 760,
        }}
      >
        La tasa de anomalías inyectada en los datos sintéticos es de{" "}
        <strong className="cifra">{porcentaje(tasaAnomalias, 2)}</strong>. El detector marca{" "}
        <strong className="cifra">{numero(stats.anomalias_detectadas)}</strong> votos, de los que{" "}
        <strong className="cifra">{numero(c.fp)}</strong> son falsos positivos: la heurística de
        concentración confunde a un candidato genuinamente popular en su región con un patrón
        fraudulento. Tiempo medio por lote entre nodos:{" "}
        <strong className="cifra">
          {(() => {
            const tiempos = Object.values(nodos)
              .filter((n) => n.nodo_id !== 0 && n.tiempo_promedio_lote > 0)
              .map((n) => n.tiempo_promedio_lote);
            return tiempos.length
              ? `${decimal(tiempos.reduce((a, b) => a + b, 0) / tiempos.length, 2)} s`
              : "—";
          })()}
        </strong>
        .
      </p>
    </>
  );
}
