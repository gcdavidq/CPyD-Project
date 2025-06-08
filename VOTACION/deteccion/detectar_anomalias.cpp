/*

*/

#include "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/VOTACION/deteccion/detectar_anomalias.hpp"
#include <omp.h>
#include <chrono>


using namespace std;

//usamos namespace deteccion para agrupar las funciones de deteccion de anomalias
namespace deteccion {

//Creamos una función del tipo ResultadoDetección que recibe un vector de Voto y devuelve un ResultadoDeteccion
ResultadoDeteccion detectarAnomaliasCPU(const vector<Voto>& votos)
{
    ResultadoDeteccion R; //Creamos un objeto ResultadoDeteccion para almacenar los resultados
    
    //Reservamos espacio para los votos válidos, tomando en cuenta el tamaño del vector de votos 
    R.validos.reserve(votos.size());

    //Tomamos una marca del tiempo antes de iniciar el procesamiento
    auto t0 = chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        vector<Voto> local_validos;
        vector<Voto> local_anomalos;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < votos.size(); ++i) {
            //Para esta simulacion, consideramos que un voto es anomalo si el hash de su DNI es divisible por MOD_HASH
            bool es_anomalo = (hash<string>{}(votos[i].dni) % MOD_HASH) == 0;
            if (es_anomalo){
                //Si el voto es anomalo, lo marcamos como tal
                Voto voto_anomalo = votos[i];
                voto_anomalo.anomalia_detectada = true; // Simulamos que se detecta la anomalía
                local_anomalos.push_back(voto_anomalo);
            }   
            //Si el voto no es anomalo, lo marcamos como valido      
            else {
                Voto voto_valido = votos[i];
                voto_valido.anomalia_detectada = false; // No es anómalo, no se detecta anomalía
                local_validos.push_back(voto_valido);
            }
        }

        #pragma omp critical
        {
            R.validos.insert(R.validos.end(),
                             local_validos.begin(), local_validos.end());
            R.anomalos.insert(R.anomalos.end(),
                              local_anomalos.begin(), local_anomalos.end());
        }
    }

    //Registramos el tiempo despues del procesamiento
    auto t1 = chrono::high_resolution_clock::now();
    R.tiempo_proceso_ms =
        chrono::duration<double, milli>(t1 - t0).count();
    return R;
}

} 
