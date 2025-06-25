#pragma once 
#include "VOTACION/common/estructura_votos.hpp"
#include "VOTACION/rendimiento/rendimiento.hpp"
#include <curl/curl.h>
#include <json/json.h>
class WebStatsSender {
public:
    WebStatsSender(const std::string& url = "http://localhost:5000");
    ~WebStatsSender();

    bool enviarEstadisticas(const Estadisticas& stats, int nodo_id = -1);
    bool enviarInfoNodo(int nodo_id, const RendimientoNodo& rendimiento);

private:
    std::string server_url;
    CURL* curl;

    struct WriteCallback {
        std::string data;
        static size_t WriteData(void* contents, size_t size, size_t nmemb, WriteCallback* userp);
    };

    std::string estadisticasToJson(const Estadisticas& stats);
};