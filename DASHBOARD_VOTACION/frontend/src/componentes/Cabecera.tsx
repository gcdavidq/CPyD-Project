import { useEffect, useState } from "react";
import type { Info } from "../tipos";
import "./Cabecera.css";

export type Vista = "resumen" | "nodos" | "mapa" | "sobre";

interface Props {
  vista: Vista;
  onVista: (v: Vista) => void;
  conectado: boolean;
  ultimoUpdate: string;
  info: Info | null;
}

const VISTAS: { clave: Vista; texto: string }[] = [
  { clave: "resumen", texto: "Resumen" },
  { clave: "nodos", texto: "Clúster" },
  { clave: "mapa", texto: "Mapa" },
  { clave: "sobre", texto: "Sobre el proyecto" },
];

type Tema = "auto" | "claro" | "oscuro";

function leerTema(): Tema {
  try {
    const t = localStorage.getItem("tema");
    if (t === "claro" || t === "oscuro" || t === "auto") return t;
  } catch {
    /* Navegación privada o almacenamiento bloqueado: se usa "auto". */
  }
  return "auto";
}

export function Cabecera({ vista, onVista, conectado, ultimoUpdate, info }: Props) {
  const [tema, setTema] = useState<Tema>(leerTema);

  useEffect(() => {
    const raiz = document.documentElement;
    if (tema === "auto") raiz.removeAttribute("data-tema");
    else raiz.setAttribute("data-tema", tema);
    try {
      localStorage.setItem("tema", tema);
    } catch {
      /* Sin persistencia; el tema sigue aplicándose en esta sesión. */
    }
  }, [tema]);

  const siguienteTema: Record<Tema, Tema> = { auto: "claro", claro: "oscuro", oscuro: "auto" };
  const iconoTema: Record<Tema, string> = { auto: "◐", claro: "☀", oscuro: "☾" };
  const nombreTema: Record<Tema, string> = { auto: "automático", claro: "claro", oscuro: "oscuro" };

  const esReplay = info?.modo === "replay";

  return (
    <header className="cabecera">
      <div className="cabecera-interior">
        <div className="cabecera-marca">
          <span className="cabecera-logo" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <rect x="2.5" y="6" width="15" height="11" rx="2" stroke="currentColor" strokeWidth="1.6" />
              <path d="M6.5 6V4.2A1.7 1.7 0 0 1 8.2 2.5h3.6A1.7 1.7 0 0 1 13.5 4.2V6" stroke="currentColor" strokeWidth="1.6" />
              <path d="M7.2 11.3l1.9 1.9 3.7-3.9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <span className="cabecera-titulo">Votación Distribuida</span>
        </div>

        <nav className="cabecera-nav" aria-label="Secciones">
          {VISTAS.map((v) => (
            <button
              key={v.clave}
              className={`cabecera-tab${vista === v.clave ? " activa" : ""}`}
              onClick={() => onVista(v.clave)}
              aria-current={vista === v.clave ? "page" : undefined}
            >
              {v.texto}
            </button>
          ))}
        </nav>

        <div className="cabecera-estado">
          {esReplay && (
            <span className="pill neutro" title="Reproducción de una ejecución real grabada del clúster">
              Reproducción
            </span>
          )}
          <span className={`cabecera-conexion${conectado ? " viva" : ""}`}>
            <i aria-hidden="true" />
            {conectado ? "En vivo" : "Sin conexión"}
          </span>
          <span className="cabecera-hora mono" title="Última actualización recibida">
            {ultimoUpdate}
          </span>
          <button
            className="cabecera-tema"
            onClick={() => setTema(siguienteTema[tema])}
            title={`Tema ${nombreTema[tema]}. Pulsa para cambiar.`}
            aria-label={`Cambiar tema. Actual: ${nombreTema[tema]}`}
          >
            {iconoTema[tema]}
          </button>
        </div>
      </div>
    </header>
  );
}
