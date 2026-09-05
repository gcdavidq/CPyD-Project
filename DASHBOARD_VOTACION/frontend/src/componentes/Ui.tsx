import type { ReactNode } from "react";

interface TarjetaProps {
  titulo: string;
  nota?: ReactNode;
  children: ReactNode;
  sinRelleno?: boolean;
  className?: string;
}

/** Contenedor estándar de un panel: título, nota opcional y cuerpo. */
export function Tarjeta({ titulo, nota, children, sinRelleno, className }: TarjetaProps) {
  return (
    <section className={`tarjeta${className ? ` ${className}` : ""}`}>
      <header className="tarjeta-cabecera">
        <h2 className="tarjeta-titulo">{titulo}</h2>
        {nota !== undefined && <span className="tarjeta-nota">{nota}</span>}
      </header>
      <div className={`tarjeta-cuerpo${sinRelleno ? " sin-relleno" : ""}`}>{children}</div>
    </section>
  );
}

interface MetricaProps {
  etiqueta: string;
  valor: ReactNode;
  pie?: ReactNode;
}

/**
 * Cifra destacada.
 *
 * Sin gráfico: cuando el dato es un único número, un número grande y bien
 * etiquetado se lee mejor que cualquier visualización.
 */
export function Metrica({ etiqueta, valor, pie }: MetricaProps) {
  return (
    <div className="metrica">
      <div className="metrica-etiqueta">{etiqueta}</div>
      <div className="metrica-valor">{valor}</div>
      {pie !== undefined && <div className="metrica-pie">{pie}</div>}
    </div>
  );
}
