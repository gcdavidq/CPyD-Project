/*
 * Constantes y configuracion global del proyecto.
 *
 * Todos los parametros se pueden ajustar con variables de entorno, sin
 * recompilar y sin tocar el codigo. Si una variable no esta definida se usa
 * el valor por defecto que aparece al lado. Ver `.env.example`.
 */
#pragma once

#include <iostream>
#include <string>
#include "VOTACION/common/estructura_votos.hpp"

namespace config {

/// Lee una variable de entorno de texto; devuelve `por_defecto` si no existe.
std::string leerEnv(const char* nombre, const std::string& por_defecto);

/// Lee una variable de entorno numerica; devuelve `por_defecto` si no existe
/// o si su contenido no es un numero valido.
int leerEnvInt(const char* nombre, int por_defecto);

/// Muestra por consola la configuracion efectiva. Lo llama el nodo maestro al
/// arrancar, para que quede constancia de con que parametros se ejecuto.
void imprimirConfiguracion();

} // namespace config


// --- Identificacion de roles MPI ---------------------------------------------
inline const int MAESTRO = 0;   ///< El rank 0 siempre coordina; el resto procesan.

// --- Parametros ajustables por entorno ---------------------------------------

/// Prefijo de los CSV de votos. A cada nodo esclavo se le anade su rank y ".csv",
/// de modo que el nodo 1 lee "<prefijo>1.csv", el nodo 2 "<prefijo>2.csv", etc.
inline const std::string ARCHIVO_ENTRADA_BASE =
    config::leerEnv("VOTACION_DATA_PREFIX", "DATA/votos_region");

/// URL del dashboard web al que el maestro envia las estadisticas por HTTP.
inline const std::string URL_DASHBOARD =
    config::leerEnv("VOTACION_DASHBOARD_URL", "http://localhost:5000");

/// Hilos OpenMP por defecto (se puede sobreescribir con el 1er argumento CLI).
inline const int NUM_HILOS_POR_DEFECTO = config::leerEnvInt("VOTACION_NUM_HILOS", 4);

/// Hilos que usa concretamente el algoritmo de deteccion de anomalias.
inline const int NUM_HILOS_PARA_ALG = config::leerEnvInt("VOTACION_HILOS_ALGORITMO", 2);

/// Votos que procesa cada lote de trabajo.
inline const int TAM_LOTE_POR_DEFECTO = config::leerEnvInt("VOTACION_TAM_LOTE", 60000);

/// Cada cuantos SEGUNDOS el maestro publica estadisticas globales.
/// Antes estaba en minutos, lo que dejaba el dashboard practicamente congelado
/// durante ejecuciones cortas.
inline const int INTERVALO_REPORTE_SEG = config::leerEnvInt("VOTACION_INTERVALO_REPORTE_SEG", 60);

/// Cada cuantos segundos el maestro evalua si hay que rebalancear la carga.
inline const int INTERVALO_CHEQUEO_BALANCEO = config::leerEnvInt("VOTACION_INTERVALO_BALANCEO_SEG", 15);

/// Segundos sin actividad tras los cuales el maestro da la ejecucion por terminada.
inline const int TIMEOUT_FINALIZACION = config::leerEnvInt("VOTACION_TIMEOUT_FIN_SEG", 30);

/// Pone a 1 para ver la traza detallada de depuracion (impacta el rendimiento).
inline const int VERBOSE = config::leerEnvInt("VOTACION_VERBOSE", 0);



// --- Etiquetas de los mensajes MPI -------------------------------------------
enum Tags {
    TAG_REPORTE_STATS     = 2,   ///< Esclavo -> Maestro: estadisticas de un lote.
    TAG_SOLICITUD_TRABAJO = 3,   ///< Esclavo -> Maestro: "dame trabajo".
    TAG_ENVIO_TRABAJO     = 4,   ///< Maestro <-> Esclavo: lote de votos.
    TAG_FINALIZAR         = 5,   ///< Maestro -> Esclavo: orden de cierre.
    TAG_CAPACIDAD_NODO    = 6,   ///< Esclavo -> Maestro: hilos y GPU disponibles.
    TAG_RESULTADO_FINAL   = 8,   ///< Esclavo -> Maestro: "he terminado del todo".
    TAG_SIN_TRABAJO       = 9,   ///< Maestro -> Esclavo: no hay nada pendiente.
    TAG_BALANCE_CARGA     = 10,  ///< Maestro -> Esclavo: cede parte de tu carga.
    TAG_RENDIMIENTO_NODO  = 11,  ///< Esclavo -> Maestro: metricas de rendimiento.
    TAG_NODO_DESOCUPADO   = 13   ///< Esclavo -> Maestro: me he quedado sin votos.
};

/// Detecta hilos disponibles y GPU (si se compilo con USE_CUDA) del nodo actual.
CapacidadNodo detectarCapacidadNodo();
