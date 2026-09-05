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

    /// Segundos transcurridos desde el inicio de la ejecucion. Se adjunta a
    /// las estadisticas para que el dashboard pueda mostrarlo; sin esto el
    /// panel enseñaba un "0.0 s" permanente.
    void fijarTiempoTranscurrido(double segundos) { tiempo_transcurrido = segundos; }

private:
    std::string server_url;
    double tiempo_transcurrido{0.0};
    CURL* curl;

    struct WriteCallback {
        std::string data;
        static size_t WriteData(void* contents, size_t size, size_t nmemb, WriteCallback* userp);
    };

    std::string estadisticasToJson(const Estadisticas& stats);
};