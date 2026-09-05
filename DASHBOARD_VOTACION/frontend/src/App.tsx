import { useCallback, useEffect, useState } from "react";
import { Cabecera, type Vista } from "./componentes/Cabecera";
import { useDatos } from "./datos/useDatos";
import { Mapa } from "./vistas/Mapa";
import { Nodos } from "./vistas/Nodos";
import { Resumen } from "./vistas/Resumen";
import { Sobre } from "./vistas/Sobre";

/** Rutas que sirve Flask. Se mantienen para que los enlaces sean compartibles. */
const RUTAS: Record<Vista, string> = {
  resumen: "/panel",
  nodos: "/nodos",
  mapa: "/mapa",
  sobre: "/",
};

function vistaDesdeRuta(ruta: string): Vista {
  const encontrada = (Object.keys(RUTAS) as Vista[]).find((v) => RUTAS[v] === ruta);
  if (encontrada) return encontrada;
  // Rutas heredadas del dashboard anterior, para no romper enlaces existentes.
  if (ruta.startsWith("/dashboard")) return "resumen";
  return "sobre";
}

export function App() {
  const datos = useDatos();
  const [vista, setVista] = useState<Vista>(() => vistaDesdeRuta(window.location.pathname));

  // Navegación con la History API: sin librería de rutas, pero los enlaces
  // siguen siendo compartibles y los botones atrás/adelante funcionan.
  useEffect(() => {
    const alNavegar = () => setVista(vistaDesdeRuta(window.location.pathname));
    window.addEventListener("popstate", alNavegar);
    return () => window.removeEventListener("popstate", alNavegar);
  }, []);

  const cambiarVista = useCallback((v: Vista) => {
    setVista(v);
    window.history.pushState({}, "", RUTAS[v]);
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }, []);

  useEffect(() => {
    const titulos: Record<Vista, string> = {
      resumen: "Resumen",
      nodos: "Clúster",
      mapa: "Mapa",
      sobre: "Sobre el proyecto",
    };
    document.title = `${titulos[vista]} · Votación Distribuida`;
  }, [vista]);

  return (
    <div className="app">
      <Cabecera
        vista={vista}
        onVista={cambiarVista}
        conectado={datos.conectado}
        ultimoUpdate={datos.stats.ultimo_update}
        info={datos.info}
      />

      <main className="contenido">
        {vista === "resumen" && <Resumen {...datos} />}
        {vista === "nodos" && <Nodos {...datos} />}
        {vista === "mapa" && <Mapa {...datos} />}
        {vista === "sobre" && <Sobre {...datos} />}
      </main>
    </div>
  );
}
