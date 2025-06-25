/*
SCRIPT PARA LAS CONSTANTES Y CONFIGURACIÓN DEL PROYECTO
*/
#pragma once

#include <iostream>
#include <string>
#include "VOTACION/common/estructura_votos.hpp"


// Constantes y configuración
inline const int MAESTRO = 0;
inline const int INTERVALO_REPORTE = 1; // minutos
inline const int SEGUNDOS_POR_MINUTO = 60;
inline int NUM_HILOS_POR_DEFECTO=4;
inline const int INTERVALO_CHEQUEO_BALANCEO = 15; // segundos
inline const int TAM_LOTE_POR_DEFECTO = 60000; // Tamaño del lote por defecto
inline const std::string ARCHIVO_ENTRADA_BASE = "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/DATA/REGIONES_100/votos_simulados100_region";
inline const int NUM_HILOS_PARA_ALG=2;
// Etiquetas para comunicación MPI
enum Tags {
    TAG_CARGA_INICIAL = 1,
    TAG_REPORTE_STATS = 2,
    TAG_SOLICITUD_TRABAJO = 3,
    TAG_ENVIO_TRABAJO = 4,
    TAG_FINALIZAR = 5,
    TAG_CAPACIDAD_NODO = 6,
    TAG_INFORME_GPU = 7,
    TAG_RESULTADO_FINAL = 8,
    TAG_SIN_TRABAJO = 9,
    TAG_BALANCE_CARGA = 10,
    TAG_RENDIMIENTO_NODO = 11,
    TAG_LOTE_VOLUNTARIO = 12,
    TAG_NODO_DESOCUPADO = 13
};

CapacidadNodo detectarCapacidadNodo();