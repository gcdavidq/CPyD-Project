# -*- coding: utf-8 -*-
"""
Convierte el CSV de bench_deteccion en un grafico de escalado en SVG.

Se dibujan dos paneles porque hay dos preguntas distintas:

  Izquierda  Cuanto tarda de verdad. Barras, directamente legibles.
  Derecha    Cuanto se acerca al ideal lineal. La distancia es el mensaje.

Un solo panel de speedup con el ideal hasta 12x deja la serie medida aplastada
contra el eje y no se distingue 1.18 de 1.24.

    python benchmarks/graficar_speedup.py benchmarks/resultados/speedup.csv

Genera version clara y oscura: GitHub elige con <picture> segun el tema del
lector. No usa dependencias externas.
"""
import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

TEMAS = {
    "claro": {
        "sufijo": "",
        "fondo": "#fcfcfb",
        "texto_primario": "#0b0b0b",
        "texto_secundario": "#52514e",
        "texto_tenue": "#78766f",
        "rejilla": "#e6e5e0",
        "eje": "#c9c8c1",
        "serie": "#2a78d6",
        "referencia": "#8a8880",
    },
    "oscuro": {
        "sufijo": "-dark",
        "fondo": "#1a1a19",
        "texto_primario": "#ffffff",
        "texto_secundario": "#c3c2b7",
        "texto_tenue": "#93918a",
        "rejilla": "#2e2e2c",
        "eje": "#464541",
        "serie": "#3987e5",
        "referencia": "#6f6e68",
    },
}

ANCHO, ALTO = 880, 408
PANEL_ANCHO = 372
SEPARACION = 68
BORDE_IZQ = 56
ARRIBA = 100          # espacio para titulo + subtitulo + titulo de panel
ABAJO = 74            # espacio para eje X + leyenda


def leer_csv(ruta):
    """Agrupa por numero de hilos y devuelve la mediana de cada grupo."""
    tiempos = defaultdict(list)
    votos = 0
    with open(ruta, newline="", encoding="utf-8") as fh:
        for fila in csv.DictReader(fh):
            tiempos[int(fila["hilos"])].append(float(fila["tiempo_ms"]))
            votos = int(fila["votos"])

    hilos = sorted(tiempos)
    medianas = {h: statistics.median(tiempos[h]) for h in hilos}
    base = medianas[hilos[0]]
    return [
        {"hilos": h,
         "ms": medianas[h],
         "speedup": base / medianas[h] if medianas[h] else 0.0}
        for h in hilos
    ], votos


def construir_svg(puntos, votos, tema):
    c = TEMAS[tema]
    partes = []
    a = partes.append

    y_base = ALTO - ABAJO
    y_techo = ARRIBA

    a('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
      'viewBox="0 0 %d %d" role="img" '
      'font-family="-apple-system,BlinkMacSystemFont,&quot;Segoe UI&quot;,'
      'Roboto,Helvetica,Arial,sans-serif">' % (ANCHO, ALTO, ANCHO, ALTO))
    a('<title>Escalado del detector de anomalias con OpenMP</title>')
    a('<rect width="%d" height="%d" fill="%s"/>' % (ANCHO, ALTO, c["fondo"]))

    # ---------------- Encabezado ----------------
    a('<text x="24" y="34" fill="%s" font-size="18" font-weight="600">'
      'Escalado del detector de anomalias con OpenMP</text>' % c["texto_primario"])
    a('<text x="24" y="55" fill="%s" font-size="12.5">'
      '%s votos por ejecucion &#183; mediana de 3 repeticiones &#183; 12 nucleos</text>'
      % (c["texto_secundario"], format(votos, ",").replace(",", "&#8239;")))

    # ================= PANEL A: tiempo por ejecucion =================
    ax0 = BORDE_IZQ
    ax1 = ax0 + PANEL_ANCHO

    a('<text x="%d" y="%d" fill="%s" font-size="13" font-weight="600">'
      'Tiempo por ejecucion</text>' % (ax0 - 32, ARRIBA - 28, c["texto_primario"]))

    max_ms = max(p["ms"] for p in puntos) * 1.18
    paso_ms = 100
    valor = 0
    while valor <= max_ms:
        y = y_base - (valor / max_ms) * (y_base - y_techo)
        a('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
          % (ax0 - 6, y, ax1, y, c["rejilla"]))
        a('<text x="%.1f" y="%.1f" fill="%s" font-size="11" text-anchor="end">%d</text>'
          % (ax0 - 12, y + 4, c["texto_tenue"], valor))
        valor += paso_ms

    a('<text x="%.1f" y="%.1f" fill="%s" font-size="11" text-anchor="end">ms</text>'
      % (ax0 - 12, y_techo - 9, c["texto_tenue"]))

    n = len(puntos)
    hueco = (ax1 - ax0) / n
    ancho_barra = min(46.0, hueco - 16)

    for i, p in enumerate(puntos):
        cx = ax0 + hueco * (i + 0.5)
        altura = (p["ms"] / max_ms) * (y_base - y_techo)
        y = y_base - altura
        # Extremo superior redondeado, anclado a la linea base.
        a('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="4" fill="%s"/>'
          % (cx - ancho_barra / 2, y, ancho_barra, altura, c["serie"]))
        a('<text x="%.1f" y="%.1f" fill="%s" font-size="11.5" font-weight="600" '
          'text-anchor="middle">%d</text>'
          % (cx, y - 8, c["texto_primario"], round(p["ms"])))
        a('<text x="%.1f" y="%.1f" fill="%s" font-size="11.5" text-anchor="middle">%d</text>'
          % (cx, y_base + 19, c["texto_tenue"], p["hilos"]))

    a('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
      % (ax0 - 6, y_base, ax1, y_base, c["eje"]))
    a('<text x="%.1f" y="%.1f" fill="%s" font-size="11.5" text-anchor="middle">'
      'Hilos OpenMP</text>' % ((ax0 + ax1) / 2, y_base + 40, c["texto_secundario"]))

    # ================= PANEL B: speedup frente al ideal =================
    bx0 = ax1 + SEPARACION
    bx1 = bx0 + PANEL_ANCHO

    a('<text x="%d" y="%d" fill="%s" font-size="13" font-weight="600">'
      'Speedup frente al ideal lineal</text>' % (bx0 - 32, ARRIBA - 28, c["texto_primario"]))

    max_hilos = puntos[-1]["hilos"]
    max_y = max_hilos * 1.08

    def bx(hilos):
        lo, hi = math.log2(puntos[0]["hilos"]), math.log2(max_hilos)
        t = 0.0 if hi == lo else (math.log2(hilos) - lo) / (hi - lo)
        # Margen interior para que las etiquetas de los extremos no se corten.
        return bx0 + 16 + t * (bx1 - bx0 - 32)

    def by(v):
        return y_base - (v / max_y) * (y_base - y_techo)

    valor = 0
    while valor <= max_y:
        y = by(valor)
        a('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
          % (bx0 - 6, y, bx1, y, c["rejilla"]))
        a('<text x="%.1f" y="%.1f" fill="%s" font-size="11" text-anchor="end">%d&#215;</text>'
          % (bx0 - 12, y + 4, c["texto_tenue"], valor))
        valor += 2

    a('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>'
      % (bx0 - 6, y_base, bx1, y_base, c["eje"]))

    # Referencia: el ideal no es una serie mas, es el marco de lectura.
    a('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2" '
      'stroke-dasharray="6 5" stroke-linecap="round"/>'
      % (bx(puntos[0]["hilos"]), by(puntos[0]["hilos"]),
         bx(max_hilos), by(max_hilos), c["referencia"]))

    camino = " ".join("%s%.1f %.1f" % ("M" if i == 0 else "L", bx(p["hilos"]), by(p["speedup"]))
                      for i, p in enumerate(puntos))
    a('<path d="%s" fill="none" stroke="%s" stroke-width="2" '
      'stroke-linejoin="round" stroke-linecap="round"/>' % (camino, c["serie"]))

    for i, p in enumerate(puntos):
        x, y = bx(p["hilos"]), by(p["speedup"])
        a('<circle cx="%.1f" cy="%.1f" r="5" fill="%s" stroke="%s" stroke-width="2"/>'
          % (x, y, c["serie"], c["fondo"]))
        a('<text x="%.1f" y="%.1f" fill="%s" font-size="11.5" text-anchor="middle">%d</text>'
          % (x, y_base + 19, c["texto_tenue"], p["hilos"]))

    # Solo se etiquetan los extremos: el primero fija la base y el ultimo es la
    # conclusion. Poner un numero sobre cada punto no anade informacion aqui.
    primero, ultimo = puntos[0], puntos[-1]
    a('<text x="%.1f" y="%.1f" fill="%s" font-size="11.5" font-weight="600" '
      'text-anchor="start">%.2f&#215;</text>'
      % (bx(primero["hilos"]) + 9, by(primero["speedup"]) + 4, c["texto_primario"],
         primero["speedup"]))
    a('<text x="%.1f" y="%.1f" fill="%s" font-size="12.5" font-weight="600" '
      'text-anchor="end">%.2f&#215;</text>'
      % (bx(ultimo["hilos"]) - 10, by(ultimo["speedup"]) - 8, c["texto_primario"],
         ultimo["speedup"]))
    a('<text x="%.1f" y="%.1f" fill="%s" font-size="11.5" text-anchor="middle">'
      'Hilos OpenMP</text>' % ((bx0 + bx1) / 2, y_base + 40, c["texto_secundario"]))

    # ---------------- Leyenda ----------------
    ly = ALTO - 16
    a('<rect x="24" y="%d" width="22" height="9" rx="3" fill="%s"/>' % (ly - 12, c["serie"]))
    a('<text x="54" y="%d" fill="%s" font-size="12">Medido</text>' % (ly - 3, c["texto_secundario"]))
    a('<line x1="126" y1="%d" x2="156" y2="%d" stroke="%s" stroke-width="2" '
      'stroke-dasharray="6 5" stroke-linecap="round"/>' % (ly - 7, ly - 7, c["referencia"]))
    a('<text x="164" y="%d" fill="%s" font-size="12">Ideal lineal</text>'
      % (ly - 3, c["texto_secundario"]))

    a("</svg>")
    return "\n".join(partes)


def main():
    parser = argparse.ArgumentParser(description="Grafica el escalado medido.")
    parser.add_argument("csv", help="CSV producido por bench_deteccion")
    parser.add_argument("--salida", default="docs/img", help="Directorio de salida")
    args = parser.parse_args()

    puntos, votos = leer_csv(args.csv)
    os.makedirs(args.salida, exist_ok=True)

    for tema, cfg in TEMAS.items():
        ruta = os.path.join(args.salida, "speedup%s.svg" % cfg["sufijo"])
        with open(ruta, "w", encoding="utf-8") as fh:
            fh.write(construir_svg(puntos, votos, tema))
        print("Escrito %s" % ruta)

    print("")
    print("Resumen:")
    for p in puntos:
        print("  %2d hilos: %8.1f ms   speedup %.2fx   eficiencia %3.0f%%"
              % (p["hilos"], p["ms"], p["speedup"], 100 * p["speedup"] / p["hilos"]))


if __name__ == "__main__":
    main()
