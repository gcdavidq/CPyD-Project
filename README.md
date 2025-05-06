# 🗳️ Sistema de Votación Electrónica Distribuida y Detección de Anomalías

Proyecto desarrollado para el curso de **Computación Paralela y Distribuida**, orientado a la simulación de un sistema de votación electrónica en el contexto peruano, utilizando **MPI (Message Passing Interface)** y **OpenMP** para la ejecución paralela en un entorno distribuido heterogéneo.

---

## 🚀 Descripción General

Este sistema simula el flujo de votos electrónicos en múltiples regiones del Perú, distribuidos entre varios nodos (laptops físicas). Cada nodo esclavo procesa de manera paralela millones de registros de votos, analiza la veracidad de los mismos y detecta anomalías como:

- Flujo excesivo de votos en un instante.
- Concentración sospechosa hacia un solo candidato.
- Repetición de votos con el mismo DNI.

El nodo maestro coordina la ejecución, recibe reportes de los esclavos y puede aplicar balanceo de carga si detecta desequilibrio en el procesamiento.

---

## 🎯 Objetivos del Proyecto

- ✅ Simular un sistema de votación distribuido realista por regiones.
- ✅ Detectar anomalías electorales mediante reglas heurísticas.
- ✅ Implementar paralelismo **distribuido** (MPI) y **local** (OpenMP).
- ✅ Diseñar un sistema adaptable a recursos heterogéneos (CPU y GPU).
- ✅ Visualizar resultados en tiempo real desde un nodo central.

---

## 🧰 Tecnologías y Herramientas Utilizadas

| Herramienta / Lenguaje | Propósito |
|------------------------|-----------|
| **C++**                | Lógica principal de procesamiento paralelo |
| **MPI (OpenMPI / MPICH)** | Comunicación entre nodos (Maestro ↔ Esclavos) |
| **OpenMP**             | Procesamiento en paralelo dentro de cada nodo |
| **Python**             | Generación de datos de votos simulados |
| **CUDA (opcional)**    | Aceleración en nodos con GPU disponibles |
| **Git / GitHub**       | Control de versiones del código y documentación |
| **CSV / JSON**         | Formatos de entrada de los votos simulados |
| **SSH + mpirun**       | Ejecución distribuida entre laptops físicas |

---



