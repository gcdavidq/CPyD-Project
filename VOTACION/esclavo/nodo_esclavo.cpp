//Script que incluye las funciones que ejecutarán los nodos esclavos
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>
#include <cstring>
#include <iomanip>
#include <ctime>
#include <random>

using namespace std;

//Estrcutura para almacenar la información de cada voto
struct Voto{
    string timestamp;
    string region;
    string dni;
    string candidato;
    bool anomalo;
    bool anomalia_detectada = false; //Para marcar las anomalias detectadas por el algoritmo
};

//Función para leer votos desde un archivo CSV
vector<Voto> leerVotos(const string& archivo){
    vector<Voto> votos;
    ifstream file(archivo);

    if (!file.is_open()){
        cerr <<"Error: No se puede abrir el archivo: " << archivo << endl;
    }

    string linea; //Saltar encabezado
    getline(file, linea);

    while(getline(file, linea)){
        stringstream ss(linea); //Separar los campos
        string campo; 
        Voto v; //Crear un voto

        getline(ss, v.timestamp, ','); //Leer el timestamp
        getline(ss, v.region, ',');
        getline(ss, v.dni, ','); //Leer la region
        getline(ss, v.candidato, ','); //Leer el dni
        getline(ss, campo, ','); //Leer el candidato
        v.anomalo = (campo == "1"); //Leer si es anomalo o no

        votos.push_back(v); //Agregar el voto al vector

    }

    return votos; //Devolver el vector de votos
}


//Funcion para simular la llegada de votos en tiempo real
void SimularLlegadaVotos(const string& archivo_region, int nodo_id, vector<Voto>& votos_totales){
    //Usar un generador de numeros aleatórios para simular llegada gradual
    random_device rd; //Variables para generar numeros aleatorios
    mt19937 gen(rd()); //Generador de numeros aleatorios
    uniform_int_distribution<> dist_tiempo(100, 500); //Entre 100 y 500 ms

    //Leer todos los votos del archivo
    vector<Voto> todos_votos = leerVotos(archivo_region); 
    cout <<"Nodo" << nodo_id <<" simulando la llegada de " << todos_votos.size() <<  " votos de la region " << archivo_region <<endl; 
    
    //Procesar votos en pequeños lotes para simular llegada en tiempo real

    const int tamaño_lote_llegada = 50;

    for (size_t i=0; i<todos_votos.size(); i+=tamaño_lote_llegada){
        //Determinar cuantos votos procesar en este lote
        size_t fin = min(i+tamaño_lote_llegada, todos_votos.size());

        //Añadir votos al contenedor central
        votos_totales.insert(votos_totales.end(), todos_votos.begin() +i, todos_votos.begin() +fin);

        //Mostrar el progreso #SOLO PARA PRUEBA
        cout << "Nodo" << nodo_id << ": Llegaron " << (fin - i) << " votos. Total acumulado: " << votos_totales.size() << endl;

        //Esperar un tiempo aleatorio para simular llegada gradual
        this_thread::sleep_for(chrono::milliseconds(dist_tiempo(gen)));

    }

}

//Funcion para probar la llegada de votos

int main(){
    vector<Voto> votos_totales;

    //Simlamos la llegada de votos desde un archivo
    SimularLlegadaVotos("/home/gianqm/Documentos/CPyD-Project/CPyD-Project/DATA/votos_simulados_region1.csv",1, votos_totales);

    //Mostrar los prmeros votos recibidos
    cout << "Primeros votos recibidos: ";
    for (int i =0; i<min((size_t)10, votos_totales.size()); i++){
        Voto v = votos_totales[i];
        cout << v.timestamp << " | " << v.region << " | "<< v.dni
            << " | "<< v.candidato << " | " << v.anomalo << endl;
    }

    return 0;
}