#include "VOTACION/estadisticas/web_stats_sender.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <curl/curl.h>
#include <json/json.h>

size_t WebStatsSender::WriteCallback::WriteData(void* contents, size_t size, size_t nmemb, WebStatsSender::WriteCallback* userp) {
    userp->data += std::string((char*)contents, size * nmemb);
    return size * nmemb;
}

WebStatsSender::WebStatsSender(const std::string& url) : server_url(url) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
}

WebStatsSender::~WebStatsSender() {
    if (curl) curl_easy_cleanup(curl);
    curl_global_cleanup();
}

std::string WebStatsSender::estadisticasToJson(const Estadisticas& stats) {
    Json::Value root;
    Json::StreamWriterBuilder builder;

    root["total_votos"] = static_cast<int>(stats.total_votos);
    root["anomalias_reales"] = static_cast<int>(stats.anomalias_reales);
    root["anomalias_detectadas"] = static_cast<int>(stats.anomalias_detectadas);
    root["falsos_positivos"] = static_cast<int>(stats.falsos_positivos);
    root["falsos_negativos"] = static_cast<int>(stats.falsos_negativos);

    for (const auto& par : stats.votos_por_region)
        root["votos_por_region"][par.first] = static_cast<int>(par.second);

    for (const auto& par : stats.votos_por_candidato)
        root["votos_por_candidato"][par.first] = static_cast<int>(par.second);

    for (const auto& region_pair : stats.votos_por_candidato_por_region)
        for (const auto& cand_pair : region_pair.second)
            root["votos_por_candidato_por_region"][region_pair.first][cand_pair.first] = static_cast<int>(cand_pair.second);

    for (const auto& region_pair : stats.anomalias_detectadas_por_region)
        for (const auto& cand_pair : region_pair.second)
            root["anomalias_por_region_candidato"][region_pair.first][cand_pair.first] = static_cast<int>(cand_pair.second);

    for (const auto& cand_pair : stats.anomalias_detectadas_por_candidato)
        for (const auto& region_pair : cand_pair.second)
            root["anomalias_por_candidato_region"][cand_pair.first][region_pair.first] = static_cast<int>(region_pair.second);

    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
    root["ultimo_update"] = ss.str();

    return Json::writeString(builder, root);
}

bool WebStatsSender::enviarEstadisticas(const Estadisticas& stats, int nodo_id) {
    if (!curl) return false;

    std::string json_data = estadisticasToJson(stats);
    std::string url = server_url + "/api/update_stats";

    WriteCallback response;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback::WriteData);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);

    if (res != CURLE_OK) {
        std::cerr << "Error CURL: " << curl_easy_strerror(res) << std::endl;
        return false;
    }

    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);

    return (response_code == 200);
}

bool WebStatsSender::enviarInfoNodo(int nodo_id, const RendimientoNodo& rendimiento) {
    if (!curl) return false;

    Json::Value root;
    Json::StreamWriterBuilder builder;

    root["nodo_id"] = nodo_id;
    root["tiempo_promedio_lote"] = rendimiento.tiempo_promedio_lote;
    root["lotes_completados"] = rendimiento.lotes_completados;
    root["tiene_gpu"] = rendimiento.tiene_gpu;
    root["carga_actual"] = rendimiento.carga_actual;
    root["numero_hilos"]=rendimiento.num_hilos;

    std::string json_data = Json::writeString(builder, root);
    std::string url = server_url + "/api/update_node";

    WriteCallback response;
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback::WriteData);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3L);

    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);

    return (res == CURLE_OK);
}
