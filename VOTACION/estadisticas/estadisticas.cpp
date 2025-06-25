#include "VOTACION/estadisticas/estadisticas.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <map>
#include <curl/curl.h>
#include <json/json.h>
#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/estadisticas/web_stats_sender.hpp"



// Función mejorada para imprimir estadísticas (mantiene la funcionalidad original + web)
void imprimirEstadisticasWeb(const Estadisticas& stats, int nodo_id, WebStatsSender& web_sender) {
    // Imprimir en consola (funcionalidad original)
    std::cout << "\n========== ESTADÍSTICAS DE PROCESAMIENTO ";
    if (nodo_id >= 0) std::cout << " NODO: " << nodo_id;
    std::cout << "======================\n";
    
    std::cout << "Total votos procesados: " << stats.total_votos << std::endl;
    std::cout << "Anomalías reales: " << stats.anomalias_reales << std::endl;
    std::cout << "Anomalías detectadas: " << stats.anomalias_detectadas << std::endl;
    std::cout << "Falsos positivos: " << stats.falsos_positivos << std::endl;
    std::cout << "Falsos negativos: " << stats.falsos_negativos << std::endl;
    
    if (stats.anomalias_reales > 0) {
        float precision = 100.0f * (stats.anomalias_detectadas - stats.falsos_positivos) / stats.anomalias_detectadas;
        float recall = 100.0f * (stats.anomalias_reales - stats.falsos_negativos) / stats.anomalias_reales;
        int verdaderos_positivos = stats.anomalias_reales - stats.falsos_negativos;
        int verdaderos_negativos = stats.total_votos - stats.anomalias_reales - stats.falsos_negativos;
        float accuracy = 100.0f * (verdaderos_positivos + verdaderos_negativos)/stats.total_votos;
        std::cout << "Precisión: " << std::fixed << std::setprecision(2) << precision << "%" << std::endl;
        std::cout << "Recall: " << std::fixed << std::setprecision(2) << recall << "%" << std::endl;
        std::cout<< "Accuracy: "<< std::fixed << std::setprecision(2)<<accuracy<<"%"<<std::endl;
    }
    
    std::cout << "\nVotos por región:\n";
    for (const auto& par : stats.votos_por_region) {
        std::cout << " " << par.first << ": " << par.second << std::endl;
    }
    
    std::cout << "\nVotos por candidato:\n";
    for (const auto& par : stats.votos_por_candidato) {
        std::cout << " " << par.first << ": " << par.second << std::endl;
    }

    std::cout << "\nVotos por candidato por región:\n";
    for (const auto& region_pair : stats.votos_por_candidato_por_region) {
        std::cout << "  Región: " << region_pair.first << std::endl;
        for (const auto& cand_pair : region_pair.second) {
            std::cout << "    Candidato " << cand_pair.first << ": " << cand_pair.second << std::endl;
        }
    }

    std::cout << "\nAnomalías detectadas por región y candidato:\n";
    for (const auto& region_pair : stats.anomalias_detectadas_por_region) {
        std::cout << "  Región: " << region_pair.first << std::endl;
        for (const auto& cand_pair : region_pair.second) {
            std::cout << "    Candidato " << cand_pair.first << ": " << cand_pair.second << " anomalías" << std::endl;
        }
    }

    std::cout << "\nAnomalías detectadas por candidato y región:\n";
    for (const auto& cand_pair : stats.anomalias_detectadas_por_candidato) {
        std::cout << "  Candidato: " << cand_pair.first << std::endl;
        for (const auto& region_pair : cand_pair.second) {
            std::cout << "    Región " << region_pair.first << ": " << region_pair.second << " anomalías" << std::endl;
        }
    }
    
    std::cout << "================================================\n";
    
    // Enviar al dashboard web
    if (!web_sender.enviarEstadisticas(stats, nodo_id)) {
        std::cout << "⚠️  No se pudo enviar estadísticas al dashboard web" << std::endl;
    }
}

void imprimirInfoNodoWeb(int nodo_id, RendimientoNodo& rendimiento, WebStatsSender& web_sender){
    std::cout<<"====ESTADISTICAS DEL RENDIMIENTO DEL NODO"<< nodo_id << "===";
    
    std::cout<<"Tiempo promedio de procesamiento"<<rendimiento.tiempo_promedio_lote<<std::endl;
    std::cout<<"Lotes completados: "<< rendimiento.lotes_completados<<std::endl;
    std::cout<<"Tiene GPU: "<<rendimiento.tiene_gpu<<std::endl;
    std::cout<<"Rendimiento nodo: "<<rendimiento.carga_actual<<std::endl;
    std::cout<<"Numeros de hilos empleados"<<rendimiento.num_hilos<<std::endl;


    //ENviar al dashboard web
    if(!web_sender.enviarInfoNodo(nodo_id, rendimiento)){
        std::cout<<" ⚠️ No se pueden enviar rendimiento al nodo web"<<std::endl; 
    }

}

using namespace std;
void imprimirEstadisticas(const Estadisticas& stats, int nodo_id){
    cout << "\n========== ESTADÍSTICAS DE PROCESAMIENTO ";
    if (nodo_id >=0) cout <<" NODO: " << nodo_id;
    cout <<"======================\n";

    cout<<"Total votos procesados: " << stats.total_votos << endl;
    cout<<"Anomalías reales: " << stats.anomalias_reales << endl;
    cout<<"Anomalías detectadas: " << stats.anomalias_detectadas << endl;
    cout<<"Falsos positivos: " << stats.falsos_positivos << endl;
    cout<<"Falsos negativos: " << stats.falsos_negativos << endl;

    if (stats.anomalias_reales > 0){
        float precision = 100.0f * (stats.anomalias_detectadas - stats.falsos_positivos) / stats.anomalias_detectadas;
        float recall = 100.0f * (stats.anomalias_reales - stats.falsos_negativos) / stats.anomalias_reales;
        cout << "Precisión: " << fixed << setprecision(2) << precision << "%" << endl;
        cout << "Recall: " << fixed << setprecision(2) << recall << "%" << endl;
        
    }

    cout << "\nVotos por región:\n";
    for (const auto& par : stats.votos_por_region) {
        cout << "  " << par.first << ": " << par.second << endl;
    }

    cout << "\nVotos por candidato:\n";
    for (const auto& par : stats.votos_por_candidato) {
        cout << "  " << par.first << ": " << par.second << endl;
    }

    cout << "\nVotos por candidato por región:\n";
    for (const auto& region_pair : stats.votos_por_candidato_por_region) {
        cout << "  Región: " << region_pair.first << endl;
        for (const auto& cand_pair : region_pair.second) {
            cout << "    Candidato " << cand_pair.first << ": " << cand_pair.second << endl;
        }
    }

    cout << "\nAnomalías detectadas por región y candidato:\n";
    for (const auto& region_pair : stats.anomalias_detectadas_por_region) {
        cout << "  Región: " << region_pair.first << endl;
        for (const auto& cand_pair : region_pair.second) {
            cout << "    Candidato " << cand_pair.first << ": " << cand_pair.second << " anomalías" << endl;
        }
    }

    cout << "\nAnomalías detectadas por candidato y región:\n";
    for (const auto& cand_pair : stats.anomalias_detectadas_por_candidato) {
        cout << "  Candidato: " << cand_pair.first << endl;
        for (const auto& region_pair : cand_pair.second) {
            cout << "    Región " << region_pair.first << ": " << region_pair.second << " anomalías" << endl;
        }
    }


    cout << "================================================\n";
}
