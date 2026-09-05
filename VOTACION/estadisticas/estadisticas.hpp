#pragma once

#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/estadisticas/web_stats_sender.hpp"

/// Vuelca las estadisticas por consola. `nodo_id` negativo = agregado global.
void imprimirEstadisticas(const Estadisticas& stats, int nodo_id = -1);

/// Igual que `imprimirEstadisticas`, y ademas las publica en el dashboard web.
void imprimirEstadisticasWeb(const Estadisticas& stats, int nodo_id, WebStatsSender& web_sender);

/// Vuelca el rendimiento de un nodo por consola y lo publica en el dashboard.
void imprimirInfoNodoWeb(int nodo_id, const RendimientoNodo& rendimiento, WebStatsSender& web_sender);
