#pragma once
#include <vector>
#include <string>
#include <map>
#include <cstring>
#include "VOTACION/common/estructura_votos.hpp"

void serializarLote(const LoteTrabajo& lote, std::vector<char>& buffer);
LoteTrabajo deserializarLote(const std::vector<char>& buffer);

void serializarEstadisticas(const Estadisticas& stats, std::vector<char>& buffer);
Estadisticas deserializarEstadisticas(const std::vector<char>& buffer);