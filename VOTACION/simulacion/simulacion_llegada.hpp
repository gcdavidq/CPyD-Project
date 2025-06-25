#pragma once
#include <string>
#include <vector>
#include <functional>
#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/common/config.hpp"
#include <mutex>


// Lee un CSV entero y devuelve todos los votos.
std::vector<Voto> leerVotos(const std::string& ruta_csv);

/**
 * Simula la llegada de votos en lotes.
 *
 * @param todos_votos   vector cargado (llámalo una sola vez al inicio)
 * @param nodo_id       solo para mensajes de log
 * @param callback      se invoca con cada lote
 * @param tam_lote      nº de votos por lote (p.ej. 150)
 */
void simularLlegadaVotos(const std::vector<Voto>& todos_votos,
                         int nodo_id,
                         std::function<void(std::vector<Voto>&&)> callback,
                         size_t TAM_LOTE_POR_DEFECTO, std::mutex& mtx_votos);

// Versión simplificada que recibe ruta y vector de votos
void simularLlegadaVotos(const std::string& ruta_csv,
                         int nodo_id,
                         std::vector<Voto>& votos_recibidos,std::mutex& mtx_votos);
