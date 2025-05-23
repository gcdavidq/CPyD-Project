/*
Script para definir las constantes y configuraciones del sistema de votación.
*/

#include <iostream>
#include <string>


// Constantes y configuración
const int MAESTRO = 0;
const int INTERVALO_REPORTE = 20; // minutos
const int SEGUNDOS_POR_MINUTO = 60;
const int NUM_HILOS_POR_DEFECTO = 4;
const int INTERVALO_CHEQUEO_BALANCEO = 30; // segundos
const std::string ARCHIVO_ENTRADA_BASE = "votos_region_";

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
    TAG_RENDIMIENTO_NODO = 11
};