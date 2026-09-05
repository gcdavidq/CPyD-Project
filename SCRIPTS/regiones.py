# -*- coding: utf-8 -*-
"""
Division territorial del simulador electoral.

El Peru se reparte en 4 macro-regiones. Cada una la procesa un nodo esclavo
distinto del cluster MPI, de modo que el numero de region coincide con el rank
del proceso que la atiende:

    rank 1 -> region 1 (Costa y Sierra Norte)
    rank 2 -> region 2 (Sierra Central)
    rank 3 -> region 3 (Sur y Oriente)
    rank 4 -> region 4 (Lima Metropolitana)

Las poblaciones son el numero aproximado de personas en edad de votar por
departamento. NOTA: este fichero antes redefinia `regiones` y `poblacion_region`
una y otra vez, por lo que en la practica solo sobrevivia el ultimo bloque
(Lima) y las otras tres regiones eran inalcanzables.
"""

# Partidos que aparecen en la papeleta simulada.
CANDIDATOS = [
    "APP",
    "APRA",
    "FUERZA POPULAR",
    "PERU LIBRE",
    "AVANZA PAIS",
]

# Poblacion electoral al 100%. Las escalas menores se obtienen multiplicando.
REGIONES = {
    1: {
        "nombre": "Costa y Sierra Norte",
        "poblacion": {
            "Piura":       2160800,
            "Lambayeque":  1400700,
            "La Libertad": 2073200,
            "San Martin":   939400,
            "Ucayali":      571200,
            "Tumbes":       263000,
            "Amazonas":     433200,
        },
    },
    2: {
        "nombre": "Sierra Central",
        "poblacion": {
            "Cajamarca":    1538900,
            "Ancash":       1262900,
            "Pasco":         293500,
            "Huanuco":       828000,
            "Junin":        1446100,
            "Huancavelica":  397700,
            "Apurimac":      467600,
        },
    },
    3: {
        "nombre": "Sur y Oriente",
        "poblacion": {
            "Arequipa":      1628100,
            "Cusco":         1396100,
            "Puno":          1362100,
            "Ayacucho":       711300,
            "Ica":            997400,
            "Moquegua":       206800,
            "Tacna":          386900,
            "Loreto":        1012000,
            "Madre de Dios":  162500,
        },
    },
    4: {
        "nombre": "Lima Metropolitana",
        "poblacion": {
            "Lima": 12411000,
        },
    },
}

# Escalas disponibles: porcentaje de la poblacion que se simula.
# 10 es suficiente para pruebas y demos; 100 genera decenas de millones de votos.
ESCALAS = (10, 50, 100)


def poblacion_de(region_id, escala=100):
    """Devuelve {departamento: electores} para una region y una escala dadas."""
    if region_id not in REGIONES:
        disponibles = ", ".join(str(r) for r in sorted(REGIONES))
        raise ValueError(
            "Region %r desconocida. Disponibles: %s" % (region_id, disponibles)
        )
    if escala not in ESCALAS:
        disponibles = ", ".join(str(e) for e in ESCALAS)
        raise ValueError(
            "Escala %r no soportada. Disponibles: %s" % (escala, disponibles)
        )

    factor = escala / 100.0
    return {
        departamento: int(habitantes * factor)
        for departamento, habitantes in REGIONES[region_id]["poblacion"].items()
    }


def nombre_de(region_id):
    """Nombre legible de la region."""
    return REGIONES[region_id]["nombre"]


def departamentos_de(region_id):
    """Lista de departamentos que componen la region."""
    return list(REGIONES[region_id]["poblacion"].keys())
