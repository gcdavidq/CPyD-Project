#pragma once 
#include "VOTACION/common/estructura_votos.hpp"
#include <curl/curl.h>
#include <json/json.h>
#include "VOTACION/estadisticas/web_stats_sender.hpp"

void imprimirEstadisticas(const Estadisticas& stats, int nodo_id=-1);


void imprimirEstadisticasWeb(const Estadisticas& stats, int nodo_id,WebStatsSender& web_sender);

void imprimirInfoNodoWeb(int nodo_id, const RendimientoNodo& rendimiento);