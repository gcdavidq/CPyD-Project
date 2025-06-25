#include <sys/resource.h>
#include <sys/time.h>
#include "VOTACION/rendimiento/rendimiento.hpp"

float obtenerUsoCPU(float tiempo_real_transcurrido) {
    struct rusage uso;
    getrusage(RUSAGE_SELF, &uso);

    float tiempo_cpu_usuario = uso.ru_utime.tv_sec + uso.ru_utime.tv_usec / 1e6;
    float tiempo_cpu_sistema = uso.ru_stime.tv_sec + uso.ru_stime.tv_usec / 1e6;
    float tiempo_cpu_total = tiempo_cpu_usuario + tiempo_cpu_sistema;

    if (tiempo_real_transcurrido > 0.0f) {
        return (tiempo_cpu_total / tiempo_real_transcurrido) * 100.0f;
    }
    return 0.0f;
}
