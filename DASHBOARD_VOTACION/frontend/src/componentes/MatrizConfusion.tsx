import { numero, porcentaje } from "../datos/calculos";
import type { Confusion } from "../tipos";
import "./MatrizConfusion.css";

interface Props {
  c: Confusion;
}

/**
 * Matriz de confusión del detector.
 *
 * Las celdas usan los colores de estado (acierto / error), no la paleta de
 * series: aquí el color significa "esto está bien" o "esto está mal", que es
 * justo para lo que están reservados.
 */
export function MatrizConfusion({ c }: Props) {
  const total = c.vp + c.fp + c.fn + c.vn;
  const cuota = (n: number) => (total > 0 ? porcentaje(n / total, 2) : "—");

  const celdas = [
    { clave: "vp", titulo: "Verdaderos positivos", corto: "VP", n: c.vp, tipo: "acierto",
      desc: "Fraude real que sí se detectó" },
    { clave: "fp", titulo: "Falsos positivos", corto: "FP", n: c.fp, tipo: "error",
      desc: "Votos legítimos marcados como fraude" },
    { clave: "fn", titulo: "Falsos negativos", corto: "FN", n: c.fn, tipo: "error",
      desc: "Fraude real que pasó desapercibido" },
    { clave: "vn", titulo: "Verdaderos negativos", corto: "VN", n: c.vn, tipo: "acierto",
      desc: "Votos legítimos correctamente ignorados" },
  ];

  return (
    <div className="confusion">
      <div className="confusion-rejilla">
        {celdas.map((celda) => (
          <div key={celda.clave} className={`confusion-celda ${celda.tipo}`} title={celda.desc}>
            <span className="confusion-corto mono">{celda.corto}</span>
            <span className="confusion-valor cifra">{numero(celda.n)}</span>
            <span className="confusion-titulo">{celda.titulo}</span>
            <span className="confusion-cuota cifra">{cuota(celda.n)}</span>
          </div>
        ))}
      </div>

      <dl className="confusion-metricas">
        <div>
          <dt>Precisión</dt>
          <dd className="cifra">{porcentaje(c.precision)}</dd>
          <span>De lo marcado, cuánto era fraude</span>
        </div>
        <div>
          <dt>Recall</dt>
          <dd className="cifra">{porcentaje(c.recall)}</dd>
          <span>Del fraude real, cuánto se pilló</span>
        </div>
        <div>
          <dt>F1</dt>
          <dd className="cifra">{porcentaje(c.f1)}</dd>
          <span>Media armónica de ambas</span>
        </div>
        <div>
          <dt>Accuracy</dt>
          <dd className="cifra">{porcentaje(c.accuracy)}</dd>
          <span>Aciertos sobre el total</span>
        </div>
      </dl>
    </div>
  );
}
