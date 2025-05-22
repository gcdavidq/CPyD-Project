/*

*/

#include "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/VOTACION/deteccion/detectar_anomalias.hpp"
#include <omp.h>
#include <chrono>


using namespace std;
namespace deteccion {

ResultadoDeteccion detectarAnomaliasCPU(const vector<Voto>& votos)
{
    ResultadoDeteccion R;
    R.validos.reserve(votos.size());

    auto t0 = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        vector<Voto> local_validos;
        vector<Voto> local_anomalos;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < votos.size(); ++i) {
            bool es_anomalo = (hash<string>{}(votos[i].dni) % MOD_HASH) == 0;
            if (es_anomalo)   local_anomalos.push_back(votos[i]);
            else              local_validos.push_back(votos[i]);
        }

        #pragma omp critical
        {
            R.validos.insert(R.validos.end(),
                             local_validos.begin(), local_validos.end());
            R.anomalos.insert(R.anomalos.end(),
                              local_anomalos.begin(), local_anomalos.end());
        }
    }

    auto t1 = chrono::high_resolution_clock::now();
    R.tiempo_proceso_ms =
        chrono::duration<double, milli>(t1 - t0).count();
    return R;
}

} // namespace deteccion
