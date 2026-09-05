import { useMemo, useState } from "react";
import { BarrasRanking } from "../componentes/BarrasRanking";
import { MapaPeru } from "../componentes/MapaPeru";
import { Tarjeta } from "../componentes/Ui";
import { buscarClave, numero, ordenar, porcentaje, totalizar } from "../datos/calculos";
import type { Datos } from "../datos/useDatos";

type Capa = "votos" | "anomalias";

export function Mapa({ stats }: Datos) {
  const [capa, setCapa] = useState<Capa>("votos");
  const [seleccion, setSeleccion] = useState<string | null>(null);

  const anomaliasPorRegion = useMemo(
    () => totalizar(stats.anomalias_por_region_candidato),
    [stats],
  );

  const valores = capa === "votos" ? stats.votos_por_region : anomaliasPorRegion;
  const unidad = capa === "votos" ? "votos" : "anomalías";

  // Datos del departamento seleccionado
  const detalle = useMemo(() => {
    if (!seleccion) return null;
    const claveVotos = buscarClave(stats.votos_por_region, seleccion);
    const claveCand = buscarClave(stats.votos_por_candidato_por_region, seleccion);
    const claveAnom = buscarClave(anomaliasPorRegion, seleccion);

    return {
      nombre: claveVotos ?? seleccion,
      votos: claveVotos ? stats.votos_por_region[claveVotos] : 0,
      anomalias: claveAnom ? anomaliasPorRegion[claveAnom] : 0,
      candidatos: claveCand ? ordenar(stats.votos_por_candidato_por_region[claveCand]) : [],
    };
  }, [seleccion, stats, anomaliasPorRegion]);

  const totalNacional = stats.total_votos;

  return (
    <div className="rejilla ancha-estrecha">
      <Tarjeta
        titulo="Distribución territorial"
        nota={
          <span className="mapa-conmutador">
            <button
              className={capa === "votos" ? "activo" : ""}
              onClick={() => setCapa("votos")}
            >
              Votos
            </button>
            <button
              className={capa === "anomalias" ? "activo" : ""}
              onClick={() => setCapa("anomalias")}
            >
              Anomalías
            </button>
          </span>
        }
      >
        <MapaPeru
          valores={valores}
          seleccion={seleccion}
          onSeleccion={setSeleccion}
          unidad={unidad}
        />
        <p style={{ marginTop: 12, marginBottom: 0, fontSize: 12, color: "var(--texto-3)", textAlign: "center" }}>
          Pulsa un departamento para ver su desglose. Los que aparecen en gris no pertenecen a
          ninguna de las cuatro macro-regiones del clúster.
        </p>
      </Tarjeta>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {detalle ? (
          <>
            <Tarjeta
              titulo={detalle.nombre}
              nota={
                <button
                  onClick={() => setSeleccion(null)}
                  style={{ border: "none", background: "none", color: "var(--acento)", padding: 0, fontSize: 12 }}
                >
                  quitar selección
                </button>
              }
            >
              <div className="rejilla dos" style={{ gap: 12 }}>
                <div>
                  <div className="metrica-etiqueta">Votos</div>
                  <div className="metrica-valor" style={{ fontSize: 22 }}>
                    {numero(detalle.votos)}
                  </div>
                  <div className="metrica-pie">
                    {totalNacional > 0 ? `${porcentaje(detalle.votos / totalNacional, 1)} del país` : "—"}
                  </div>
                </div>
                <div>
                  <div className="metrica-etiqueta">Anomalías</div>
                  <div className="metrica-valor" style={{ fontSize: 22 }}>
                    {numero(detalle.anomalias)}
                  </div>
                  <div className="metrica-pie">
                    {detalle.votos > 0 ? `${porcentaje(detalle.anomalias / detalle.votos, 1)} de sus votos` : "—"}
                  </div>
                </div>
              </div>
            </Tarjeta>

            <Tarjeta titulo="Reparto por candidato" nota={detalle.nombre}>
              <BarrasRanking datos={detalle.candidatos} limite={6} />
            </Tarjeta>
          </>
        ) : (
          <>
            <Tarjeta titulo="Ranking de departamentos" nota={unidad}>
              <BarrasRanking
                datos={ordenar(valores)}
                limite={10}
                etiquetaResto="Otros"
                seleccion={seleccion}
                onSeleccion={setSeleccion}
              />
            </Tarjeta>

            <Tarjeta titulo="Reparto nacional por candidato" nota="todos los departamentos">
              <BarrasRanking datos={ordenar(stats.votos_por_candidato)} limite={6} />
            </Tarjeta>
          </>
        )}
      </div>
    </div>
  );
}
