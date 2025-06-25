/* 
SCRIPT QUE INCLUYE TODAS LAS ESTRUCTURAS DE DATOS NECESARIAS PARA EL PROYECTO
*/
#pragma once
#include <string>
#include <vector>
#include <map>
#include <chrono>

// Estructura para almacenar la información de cada voto
struct Voto {
    std::string timestamp;
    std::string region;
    std::string dni;
    std::string candidato;
    bool  anomalo{false};
    bool  anomalia_detectada{false};
    int tipo_anomalia; //NUEVO
};

// Estructura para almacenar un lote de trabajo
struct LoteTrabajo {
    std::vector<Voto> votos;
    int         id_lote{0};
    std::string inicio_timestamp;
    std::string fin_timestamp;
};

// Estructura para almacenar estadísticas de procesamiento
struct Estadisticas {
    int total_votos{0};
    int anomalias_reales{0};
    int anomalias_detectadas{0};
    int falsos_positivos{0};
    int falsos_negativos{0};
    std::map<std::string,int> votos_por_region;
    std::map<std::string,int> votos_por_candidato;

    //nuevas
    std::map<std::string, std::map<std::string, int>> votos_por_candidato_por_region;
    std::map<std::string, std::map<std::string, int>> anomalias_detectadas_por_region;
    std::map<std::string, std::map<std::string, int>> anomalias_detectadas_por_candidato;


    // funcion miembro para combinar estadísticas
    void combinar(const Estadisticas& otra) {
        total_votos          += otra.total_votos;
        anomalias_reales     += otra.anomalias_reales;
        anomalias_detectadas += otra.anomalias_detectadas;
        falsos_positivos     += otra.falsos_positivos;
        falsos_negativos     += otra.falsos_negativos;

        for (const auto& p : otra.votos_por_region)
            votos_por_region[p.first] += p.second;

        for (const auto& p : otra.votos_por_candidato)
            votos_por_candidato[p.first] += p.second;

        // Nuevos mapas anidados
        for (const auto& region_par : otra.votos_por_candidato_por_region) {
            const std::string& region = region_par.first;
            for (const auto& cand_par : region_par.second) {
                votos_por_candidato_por_region[region][cand_par.first] += cand_par.second;
            }
        }

        for (const auto& region_par : otra.anomalias_detectadas_por_region) {
            const std::string& region = region_par.first;
            for (const auto& cand_par : region_par.second) {
                anomalias_detectadas_por_region[region][cand_par.first] += cand_par.second;
            }
        }

        for (const auto& candidato_par : otra.anomalias_detectadas_por_candidato) {
            const std::string& candidato = candidato_par.first;
            for (const auto& region_par : candidato_par.second) {
                anomalias_detectadas_por_candidato[candidato][region_par.first] += region_par.second;
            }
        }
    }
};

// Estructura para almacenar la capacidad de un nodo
struct CapacidadNodo {
    int   num_hilos{0};
    bool  tiene_gpu{false};
    float rendimiento_relativo{1.0f};
    int   gpu_memoria_mb{0};
    char gpu_modelo[128];
    float velocidad_procesamiento{0.0f};  // lotes/s
    int   lotes_pendientes{0};

};

// Estructura para almacenar el rendimiento de un nodo
struct RendimientoNodo {
    int   nodo_id{0};
    float tiempo_promedio_lote{0.0f};  // s
    int   lotes_completados{0};
    int   lotes_asignados{0};
    bool  tiene_gpu{false};
    float carga_actual{0.0f};          // %
    std::chrono::time_point<std::chrono::system_clock> ultimo_reporte;
    double tiempo_comunicacion_mpi{0};
    int num_hilos {0};
    //double tiempo_sincronizacion_mpi{0};
};