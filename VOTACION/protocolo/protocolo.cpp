#include <VOTACION/protocolo/protocolo.hpp>


void serializarLote(const LoteTrabajo& lote, std::vector<char>& buffer) {
    // Calcular el tamaño total necesario para la serializacion
    size_t tamano_total = sizeof(int) * 2; //id del lote + cantidad de votos
    tamano_total += lote.inicio_timestamp.size() + 1;
    tamano_total += lote.fin_timestamp.size() + 1;

    //Por cada voto dentro del lote, se suma el tamaño de sus campos
    for (const auto& voto : lote.votos) {
        tamano_total += voto.timestamp.size() + 1;
        tamano_total += voto.region.size() + 1;
        tamano_total += voto.dni.size() + 1;
        tamano_total += voto.candidato.size() + 1;
        tamano_total += sizeof(bool) * 2;
    }

    //Redimencionamos el tamaño del buffer y obtenemos un puntero al inicio del buffer
    buffer.resize(tamano_total);
    char* ptr = buffer.data(); //.data para obtener un puntero al inicio del vector de caracteres

    //memcpy es una funcion que copia bloques de memoria, lo estamos usando para copiar los datos del lote al buffer
    // se emplea cuando el dato es binario y no se puede copiar directamente como string
    //ptr es un puntero que se va moviendo a medida que copiamos los datos

    memcpy(ptr, &lote.id_lote, sizeof(int)); ptr += sizeof(int); //destino, origen, tamaño
    int tam = lote.votos.size();
    memcpy(ptr, &tam, sizeof(int)); ptr += sizeof(int);
    memcpy(ptr, lote.inicio_timestamp.c_str(), lote.inicio_timestamp.size() + 1); ptr += lote.inicio_timestamp.size() + 1;
    memcpy(ptr, lote.fin_timestamp.c_str(), lote.fin_timestamp.size() + 1); ptr += lote.fin_timestamp.size() + 1;

    // nos aseguramos que sea solo lectura y no se copie cada elemento de la estructura voto
    for (const auto& voto : lote.votos) {
        memcpy(ptr, voto.timestamp.c_str(), voto.timestamp.size() + 1); 
        ptr += voto.timestamp.size() + 1;
        memcpy(ptr, voto.region.c_str(), voto.region.size() + 1); 
        ptr += voto.region.size() + 1;
        memcpy(ptr, voto.dni.c_str(), voto.dni.size() + 1); 
        ptr += voto.dni.size() + 1;
        memcpy(ptr, voto.candidato.c_str(), voto.candidato.size() + 1); 
        ptr += voto.candidato.size() + 1;
        memcpy(ptr, &voto.anomalo, sizeof(bool)); 
        ptr += sizeof(bool);
        memcpy(ptr, &voto.anomalia_detectada, sizeof(bool)); 
        ptr += sizeof(bool);
    }
}

LoteTrabajo deserializarLote(const std::vector<char>& buffer) {
    LoteTrabajo lote;
    const char* ptr = buffer.data();
    //ptr recorrerá el buffer de datos, apuntando a la posición actual

    memcpy(&lote.id_lote, ptr, sizeof(int)); ptr += sizeof(int);
    int tam; 
    memcpy(&tam, ptr, sizeof(int)); 
    ptr += sizeof(int);
    lote.inicio_timestamp = ptr; 
    ptr += lote.inicio_timestamp.size() + 1;
    lote.fin_timestamp = ptr; 
    ptr += lote.fin_timestamp.size() + 1;

    lote.votos.resize(tam);
    for (int i = 0; i < tam; i++) {
        lote.votos[i].timestamp = ptr; 
        ptr += lote.votos[i].timestamp.size() + 1;
        lote.votos[i].region = ptr; 
        ptr += lote.votos[i].region.size() + 1;
        lote.votos[i].dni = ptr; ptr += lote.votos[i].dni.size() + 1;
        lote.votos[i].candidato = ptr; 
        ptr += lote.votos[i].candidato.size() + 1;
        memcpy(&lote.votos[i].anomalo, ptr, sizeof(bool)); 
        ptr += sizeof(bool);
        memcpy(&lote.votos[i].anomalia_detectada, ptr, sizeof(bool)); 
        ptr += sizeof(bool);
    }
    return lote;
}

void serializarEstadisticas(const Estadisticas& stats, std::vector<char>& buffer) {
    size_t tamano_total = sizeof(int) * 5;
    tamano_total += sizeof(int);
    for (const auto& par : stats.votos_por_region) {
        tamano_total += par.first.size() + 1 + sizeof(int);
    }
    tamano_total += sizeof(int);
    for (const auto& par : stats.votos_por_candidato) {
        tamano_total += par.first.size() + 1 + sizeof(int);
    }

    buffer.resize(tamano_total);
    char* ptr = buffer.data();

    memcpy(ptr, &stats.total_votos, sizeof(int)); 
    ptr += sizeof(int);
    memcpy(ptr, &stats.anomalias_reales, sizeof(int)); 
    ptr += sizeof(int);
    memcpy(ptr, &stats.anomalias_detectadas, sizeof(int)); 
    ptr += sizeof(int);
    memcpy(ptr, &stats.falsos_positivos, sizeof(int)); 
    ptr += sizeof(int);
    memcpy(ptr, &stats.falsos_negativos, sizeof(int)); 
    ptr += sizeof(int);

    int nreg = stats.votos_por_region.size();
    memcpy(ptr, &nreg, sizeof(int)); ptr += sizeof(int);
    for (const auto& par : stats.votos_por_region) {
        memcpy(ptr, par.first.c_str(), par.first.size() + 1); 
        ptr += par.first.size() + 1;
        memcpy(ptr, &par.second, sizeof(int)); ptr += sizeof(int);
    }

    int ncand = stats.votos_por_candidato.size();
    memcpy(ptr, &ncand, sizeof(int)); ptr += sizeof(int);
    for (const auto& par : stats.votos_por_candidato) {
        memcpy(ptr, par.first.c_str(), par.first.size() + 1); 
        ptr += par.first.size() + 1;
        memcpy(ptr, &par.second, sizeof(int)); ptr += sizeof(int);
    }
}

Estadisticas deserializarEstadisticas(const std::vector<char>& buffer) {
    Estadisticas stats;
    const char* ptr = buffer.data();

    memcpy(&stats.total_votos, ptr, sizeof(int)); ptr += sizeof(int);
    memcpy(&stats.anomalias_reales, ptr, sizeof(int)); ptr += sizeof(int);
    memcpy(&stats.anomalias_detectadas, ptr, sizeof(int)); ptr += sizeof(int);
    memcpy(&stats.falsos_positivos, ptr, sizeof(int)); ptr += sizeof(int);
    memcpy(&stats.falsos_negativos, ptr, sizeof(int)); ptr += sizeof(int);

    int nreg;
    memcpy(&nreg, ptr, sizeof(int)); ptr += sizeof(int);
    for (int i = 0; i < nreg; i++) {
        std::string region = ptr; ptr += region.size() + 1;
        int cont; memcpy(&cont, ptr, sizeof(int)); ptr += sizeof(int);
        stats.votos_por_region[region] = cont;
    }

    int ncand;
    memcpy(&ncand, ptr, sizeof(int)); ptr += sizeof(int);
    for (int i = 0; i < ncand; i++) {
        std::string cand = ptr; ptr += cand.size() + 1;
        int cont; memcpy(&cont, ptr, sizeof(int)); ptr += sizeof(int);
        stats.votos_por_candidato[cand] = cont;
    }

    return stats;
}
