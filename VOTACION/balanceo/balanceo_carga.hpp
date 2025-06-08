#ifndef BALANCEO_CARGA_HPP
#define BALANCEO_CARGA_HPP

#include <vector>
#include <map>
#include <queue>

#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/common/config.hpp"
#include "VOTACION/protocolo/protocolo.hpp"
#include "estado_balanceo.hpp"

void balanceoCarga(std::vector<RendimientoNodo>& rendimiento_nodos,
                    std:: queue<LoteTrabajo>& cola_trabajo,
                    int num_nodos);

#endif