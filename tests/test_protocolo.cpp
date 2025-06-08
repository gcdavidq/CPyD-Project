/*
Script para probar la funcionalidad de la serialización y deserialización de lotes y estadisticas
*/
#include <iostream>
#include "VOTACION/protocolo/protocolo.hpp"

using namespace std;
int main(){
    //Crear un lote de trabajo de ejemplo
    LoteTrabajo lote;
    lote.id_lote = 1;
    lote.inicio_timestamp = "2025-06-05 12:00:00";
    lote.fin_timestamp = "2025-06-05 12:05:00";
    lote.votos.push_back({"2025-06-05 12:00:01", "Region1", "70857114", "APRA", false, false});
    lote.votos.push_back({"2025-06-05 12:00:02", "Region2", "70857115", "Fuerza Popular", false, false});
    lote.votos.push_back({"2025-06-05 12:00:03", "Region1", "70857116", "APRA", true, false});
    lote.votos.push_back({"2025-06-05 12:00:04", "Region2", "70857117", "Fuerza Popular", false, true});
    lote.votos.push_back({"2025-06-05 12:00:05", "Region1", "70857118", "APRA", true, true});

    //Serializar el lote
    vector<char> buffer_lote;
    serializarLote(lote, buffer_lote); 
    cout <<"Lote serializado correctamente. Tamaño del buffer: " << buffer_lote.size() << "bytes" << endl;

    //Deserializar el lote
    LoteTrabajo lote_deeserializado = deserializarLote(buffer_lote);
    cout << "Lote deserializado correctamente. ID: " << lote_deeserializado.id_lote << endl;
    cout << "Inicio: " << lote_deeserializado.inicio_timestamp << ", Fin: " << lote_deeserializado.fin_timestamp << endl;
    for (const auto& voto : lote_deeserializado.votos) {
        cout << "Voto: " << voto.timestamp << ", Region: " << voto.region 
             << ", DNI: " << voto.dni << ", Candidato: " << voto.candidato
             << ", Anómalo: " << (voto.anomalo ? "Sí" : "No")
             << ", Anomalia Detectada: " << (voto.anomalia_detectada ? "Sí" : "No") << endl;
    }

    // Crear estadísticas de ejemplo
    Estadisticas stats;
    stats.total_votos = 1000;
    stats.anomalias_reales = 50;
    stats.anomalias_detectadas = 45;
    stats.falsos_positivos = 5;
    stats.falsos_negativos = 5;
    stats.votos_por_region["Region1"] = 600;
    stats.votos_por_region["Region2"] = 400;
    stats.votos_por_candidato["APRA"] = 550;
    stats.votos_por_candidato["Fuerza Popular"] = 450;  

    // Serializar las estadísticas
    vector<char> buffer_stats;  
    serializarEstadisticas(stats, buffer_stats);
    cout << "Estadísticas serializadas correctamente. Tamaño del buffer: " << buffer_stats.size() << " bytes" << endl;

    // Deserializar las estadísticas
    Estadisticas stats_deserializadas = deserializarEstadisticas(buffer_stats);
    cout << "Estadísticas deserializadas correctamente." << endl;
    cout << "Total Votos: " << stats_deserializadas.total_votos << endl;
    cout << "Anomalías Reales: " << stats_deserializadas.anomalias_reales << endl;
    cout << "Anomalías Detectadas: " << stats_deserializadas.anomalias_detectadas << endl;  
    cout << "Falsos Positivos: " << stats_deserializadas.falsos_positivos << endl;
    cout << "Falsos Negativos: " << stats_deserializadas.falsos_negativos << endl;
    cout << "Votos por Región:" << endl;
    for (const auto& par : stats_deserializadas.votos_por_region) {
        cout << "  " << par.first << ": " << par.second << endl;
    }
    cout << "Votos por Candidato:" << endl;
    for (const auto& par : stats_deserializadas.votos_por_candidato) {
        cout << "  " << par.first << ": " << par.second << endl;

    }
    return 0;
}

