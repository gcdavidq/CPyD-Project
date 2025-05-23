/*
Script donde se encuentran las funciones del nodo maestro
*/
#include <iostream>
#include <vector>
#include <thread>
#include <omp.h>
#include<mpi.h>
#include "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/VOTACION/simulacion/simulacion_llegada.hpp"
#include "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/VOTACION/deteccion/detectar_anomalias.hpp"
#include "/home/gianqm/Documentos/CPyD-Project/CPyD-Project/VOTACION/common/config.hpp"

using namespace std;
//Funcion para detectar la capacidades de un nodo
CapacidadNodo detectarCapacidadNodo(){
    CapacidadNodo capacidad;
    capacidad.num_hilos = omp_get_max_threads(); // Detectar el número de hilos disponibles
    capacidad.tiene_gpu=false; // Inicializar como falso 
    capacidad.rendimiento_relativo = 1.0; // Inicializar rendimiento relativo
    capacidad.gpu_memoria_mb =0; // Inicializar memoria GPU
    capacidad.gpu_modelo = "ninguno";
    capacidad.velocidad_procesamiento = 0.0;
    capacidad.lotes_pendientes = 0; 

    #ifdef USE_CUDA
     int deviceCount = 0; // Contar dispositivos CUDA disponibles
     cudaError_t error = cudaGetDeviceCount(&deviceCount); // Obtener el número de dispositivos CUDA
     if (error==cudaSuccess && deviceCount>0){
        capacidad.tiene_gpu = true;
        
        //Obtener información del primer dispositivo
         cudaDeviceProp deviceProp;
         cudaGetDeviceProperties(&deviceProp, 0); // Obtener propiedades del dispositivo
         capacidad.gpu_memoria_mb = deviceProp.totalGlobalMem / (1024 * 1024); //Convertir a MB
         capacidad.gpu_modelo=deviceProp.name; //Obtener el nombre del modelo

         // Estimar rendimiento relativo basado en memoria y núcleos
         capacidad.rendimiento_relativo = capacidad.tiene_gpu ?
            2.0 + (deviceProp.multiProcessorCount / 20.0): 1.0; // Estimación de rendimiento
     }

     #endif

     return capacidad; 
    


}

void ejecutarNodoMaestro(int num_nodos){

    cout<<"Iniciando nodo maestro con "<<num_nodos-1<< "nodos esclavos" <<endl;

    //Detectar capacidades del nodo maestro
    CapacidadNodo capacidad_maestro = detectarCapacidadNodo();

    cout << "Capacidad del maestro: "<<capacidad_maestro.num_hilos << "hilos, GPU: "
        << (capacidad_maestro.tiene_gpu ? "Si" : "NO") <<endl;

    //EStructura para gestionar los nodos
    vector <CapacidadNodo> capacidades_nodos (num_nodos); //creamos un vector de capacidades
    capacidades_nodos[MAESTRO] = capacidad_maestro;
    
    vector<RendimientoNodo> rendimiento_nodos(num_nodos); //creamos un vector de rendimiento
    for (int i =0; i<num_nodos; i++){
        rendimiento_nodos[i].nodo_id = i; //Asignamos el id del nodo
        rendimiento_nodos[i].tiempo_promedio_lote = 0.0f; //inicializamos el tiempo promedio
        rendimiento_nodos[i].lotes_completados = 0; //inicializamos los lotes completados
        rendimiento_nodos[i].lotes_asignados = 0; //inicializamos los lotes asignados
        rendimiento_nodos[i].tiene_gpu = false; //inicializamos la GPU
        rendimiento_nodos[i].carga_actual = 0.0f; //inicializamos la carga actual
        rendimiento_nodos[i].ultimo_reporte = chrono::system_clock::now(); //inicializamos el tiempo de reporte
        
    }

    //Recibir informaicon sobre capacidades de los nodos
    for (int i =1; i<num_nodos; i++){
        MPI_Status estado; //MPI_Status es una estructura que contiene información sobre el estado de la comunicación

        vector<char> buffer(sizeof(CapacidadNodo)); //Creamos un buffer para recibir la información


    }



}