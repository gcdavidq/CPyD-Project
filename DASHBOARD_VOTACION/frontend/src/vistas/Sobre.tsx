import { Tarjeta } from "../componentes/Ui";
import { numero } from "../datos/calculos";
import type { Datos } from "../datos/useDatos";
import "./Sobre.css";

const REPO = "https://github.com/gcdavidq/CPyD-Project";

export function Sobre({ stats, info }: Datos) {
  const esReplay = info?.modo === "replay";

  return (
    <div className="sobre">
      <section className="sobre-intro">
        <h1>Un clúster que cuenta votos y busca fraude en paralelo</h1>
        <p>
          Este panel monitoriza un sistema distribuido escrito en C++ que reparte una jornada
          electoral simulada entre varias máquinas. Cada nodo procesa su parte del padrón con{" "}
          <strong>OpenMP</strong>, se comunica con el resto por <strong>MPI</strong>, y un nodo
          maestro agrega los resultados y redistribuye la carga cuando detecta que un nodo se
          está quedando atrás.
        </p>
        <div className="sobre-acciones">
          <a className="sobre-boton" href={REPO} target="_blank" rel="noreferrer">
            Ver el código en GitHub
          </a>
          <span className="sobre-nota">
            Ejercicio académico · datos 100 % sintéticos
          </span>
        </div>
      </section>

      <div className="rejilla tres" style={{ marginBottom: 16 }}>
        <Tarjeta titulo="Qué resuelve">
          <p className="sobre-parrafo">
            Contar millones de votos es trivial. Lo difícil es hacerlo <em>repartido</em> entre
            máquinas de potencia distinta sin que las lentas frenen a las rápidas, y detectar
            patrones anómalos mientras los datos siguen llegando.
          </p>
        </Tarjeta>
        <Tarjeta titulo="Cómo detecta el fraude">
          <ul className="sobre-lista">
            <li>
              <strong>DNI duplicado</strong> — la misma persona votando varias veces
            </li>
            <li>
              <strong>Concentración</strong> — un candidato acaparando una región por encima de
              la media más dos desviaciones típicas
            </li>
            <li>
              <strong>Flujo excesivo</strong> — una avalancha de votos en el mismo minuto
            </li>
          </ul>
        </Tarjeta>
        <Tarjeta titulo="Dos niveles de paralelismo">
          <p className="sobre-parrafo">
            <strong>MPI</strong> reparte las macro-regiones entre nodos y transporta los lotes
            serializados a mano sobre búferes binarios. Dentro de cada nodo, <strong>OpenMP</strong>{" "}
            clasifica los votos en paralelo; si hay GPU, un kernel <strong>CUDA</strong> toma el
            relevo.
          </p>
        </Tarjeta>
      </div>

      <Tarjeta titulo="Lo que muestra este panel" nota="datos reales de una ejecución">
        <div className="sobre-hechos">
          <div>
            <span className="sobre-cifra cifra">{numero(stats.total_votos)}</span>
            <span>votos procesados en esta ejecución</span>
          </div>
          <div>
            <span className="sobre-cifra cifra">{Object.keys(stats.votos_por_region).length}</span>
            <span>departamentos del Perú</span>
          </div>
          <div>
            <span className="sobre-cifra cifra">{Object.keys(stats.votos_por_candidato).length}</span>
            <span>agrupaciones políticas simuladas</span>
          </div>
          <div>
            <span className="sobre-cifra cifra">{stats.nodos_activos}</span>
            <span>procesos MPI coordinados</span>
          </div>
        </div>
      </Tarjeta>

      {esReplay && (
        <div className="sobre-aviso">
          <strong>Sobre esta demo.</strong> No hay un clúster MPI corriendo detrás de esta
          página: un despliegue web gratuito no puede levantar varios procesos con red entre
          ellos. Lo que ves es la <em>reproducción</em> de una ejecución real, grabada evento a
          evento con sus marcas de tiempo. Los números son auténticos, solo que diferidos. Para
          ejecutar el sistema de verdad basta con{" "}
          <code className="mono">docker compose up</code> en el repositorio.
        </div>
      )}

      <div className="sobre-aviso sobre-aviso-legal">
        <strong>Aviso.</strong> Proyecto del curso de Computación Paralela y Distribuida. Todos
        los datos son generados por un script y no proceden de ninguna elección real. Los
        nombres de las agrupaciones se usan únicamente para dar verosimilitud a la simulación.
        Este sistema es un ejercicio de computación paralela: no es software electoral apto para
        uso real.
      </div>
    </div>
  );
}
