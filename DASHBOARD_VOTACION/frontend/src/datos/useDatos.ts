import { useEffect, useRef, useState } from "react";
import { io, type Socket } from "socket.io-client";
import {
  ESTADISTICAS_VACIAS,
  type Estadisticas,
  type Info,
  type Muestra,
  type Nodos,
} from "../tipos";

/** Cuántos puntos guarda la serie temporal antes de empezar a descartar. */
const MAX_MUESTRAS = 240;

export interface Datos {
  stats: Estadisticas;
  nodos: Nodos;
  info: Info | null;
  conectado: boolean;
  serie: Muestra[];
}

/**
 * Única fuente de datos de la aplicación.
 *
 * Hace una carga inicial por HTTP (para que la página tenga contenido aunque
 * el socket tarde) y a partir de ahí escucha los eventos que emite Flask.
 * El histórico de la serie viene del servidor; los puntos posteriores se van
 * añadiendo aquí conforme llegan las actualizaciones.
 */
export function useDatos(): Datos {
  const [stats, setStats] = useState<Estadisticas>(ESTADISTICAS_VACIAS);
  const [nodos, setNodos] = useState<Nodos>({});
  const [info, setInfo] = useState<Info | null>(null);
  const [conectado, setConectado] = useState(false);
  const [serie, setSerie] = useState<Muestra[]>([]);

  const inicio = useRef<number>(Date.now());
  const ultimoTotal = useRef<number>(-1);

  useEffect(() => {
    let vivo = true;

    // Carga inicial: si el socket falla, al menos hay datos en pantalla.
    const cargar = async (ruta: string, aplicar: (d: unknown) => void) => {
      try {
        const r = await fetch(ruta);
        if (!r.ok) return;
        const d = await r.json();
        if (vivo) aplicar(d);
      } catch {
        /* Sin conexión: el indicador de estado ya lo refleja. */
      }
    };

    void cargar("/api/estadisticas", (d) => {
      const s = d as Estadisticas;
      setStats(s);
      ultimoTotal.current = s.total_votos;
    });

    // El historial lo guarda el servidor, así que la gráfica tiene pasado
    // desde el primer instante aunque se abra la página a mitad de ejecución.
    void cargar("/api/serie", (d) => {
      const crudos = d as Muestra[];
      if (!Array.isArray(crudos) || crudos.length === 0) return;

      // El servidor cuenta en segundos y aquí se trabaja en milisegundos.
      const puntos = crudos
        .slice(-MAX_MUESTRAS)
        .map((p) => ({ ...p, t: p.t * 1000 }));

      setSerie(puntos);
      // El reloj local se alinea con el del servidor para que los puntos que
      // lleguen después encajen con los que ya están dibujados.
      inicio.current = Date.now() - puntos[puntos.length - 1].t;
    });
    void cargar("/api/nodos", (d) => setNodos(d as Nodos));
    void cargar("/api/info", (d) => setInfo(d as Info));

    const socket: Socket = io({
      // El backend usa el modo "threading" de Flask-SocketIO, que no ofrece
      // WebSocket real: forzar el transporte fallaría. Con polling basta para
      // un refresco cada pocos segundos.
      transports: ["polling", "websocket"],
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });

    socket.on("connect", () => vivo && setConectado(true));
    socket.on("disconnect", () => vivo && setConectado(false));
    socket.on("connect_error", () => vivo && setConectado(false));

    socket.on("estadisticas_update", (d: Estadisticas) => {
      if (!vivo) return;
      setStats(d);

      // Solo se anota un punto cuando el total cambia: el replay reenvía el
      // mismo estado varias veces y no aporta nada repetirlo en la gráfica.
      if (d.total_votos !== ultimoTotal.current) {
        ultimoTotal.current = d.total_votos;
        setSerie((prev) => {
          const siguiente = [
            ...prev,
            {
              t: Date.now() - inicio.current,
              votos: d.total_votos,
              anomalias: d.anomalias_detectadas,
            },
          ];
          // Un ciclo de replay reinicia el contador: al detectarlo se empieza
          // una serie nueva en lugar de dibujar una caída falsa.
          const ultimo = siguiente[siguiente.length - 1];
          const penultimo = siguiente[siguiente.length - 2];
          if (penultimo && ultimo.votos < penultimo.votos) {
            inicio.current = Date.now();
            return [{ t: 0, votos: ultimo.votos, anomalias: ultimo.anomalias }];
          }
          return siguiente.slice(-MAX_MUESTRAS);
        });
      }
    });

    socket.on("nodos_update", (d: Nodos) => vivo && setNodos(d));

    return () => {
      vivo = false;
      socket.close();
    };
  }, []);

  return { stats, nodos, info, conectado, serie };
}
