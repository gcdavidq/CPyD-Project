/*
Script para probar el procesamiento de lotes de votos.
*/
#include "VOTACION/procesamiento/procesar_lote.hpp"
#include <iostream>

using namespace std; 

int main(){
    // Crear un lote de trabajo con algunos votos de ejemplo
    LoteTrabajo lote;
    lote.id_lote =1;
    lote.inicio_timestamp = "2025-06-08 12:00:00";
    lote.fin_timestamp = "2025-06-08 12:01:00";
    lote.votos = {
        {"2025-06-08 12:00:01", "Region1", "12345678", "Candidato1", false},
        {"2025-06-08 12:00:02", "Region1", "87654321", "Candidato2", false},
        {"2025-06-08 12:00:03", "Region2", "12345678", "Candidato1", true},
        {"2025-06-08 12:00:04", "Region2", "23456789", "Candidato3", false},
        {"2025-06-08 12:00:05", "Region3", "34567890", "Candidato2", true},
        {"2025-06-08 12:00:06", "Region3", "45678901", "Candidato1", false},
        {"2025-06-08 12:00:07", "Region1", "56789012", "Candidato3", true},
        {"2025-06-08 12:00:08", "Region2", "67890123", "Candidato2", false},
        {"2025-06-08 12:00:09", "Region3", "78901234", "Candidato1", false},
        {"2025-06-08 12:00:10", "Region1", "89012345", "Candidato3", true},
        {"2025-06-08 12:00:11", "Region2", "90123456", "Candidato2", false},
        {"2025-06-08 12:00:12", "Region3", "01234567", "Candidato1", true}
    };
    // Procesar el lote con la CPU
    Estadisticas stats = procesarLote(lote, false); // false indica que no se usa GPU
    // Mostrar estadísticas del procesamiento
    cout << "Total de votos procesados: " << stats.total_votos << endl;
    cout << "Anomalías reales: " << stats.anomalias_reales << endl;
    cout << "Anomalías detectadas: " << stats.anomalias_detectadas << endl;
    cout << "Falsos positivos: " << stats.falsos_positivos << endl;
    cout << "Falsos negativos: " << stats.falsos_negativos << endl;
    cout << "Votos por región:" << endl;
    for (const auto& p : stats.votos_por_region) {
        cout << "  " << p.first << ": " << p.second << endl;
    }
    cout << "Votos por candidato:" << endl;
    for (const auto& p : stats.votos_por_candidato) {
        cout << "  " << p.first << ": " << p.second << endl;
    }
    
    return 0;

}