# Guía de despliegue

Cómo poner la demo pública en marcha. Son unos 15 minutos y **no hace falta
tarjeta de crédito**.

---

## Qué se despliega exactamente

Solo el **dashboard**, en modo *replay*: reproduce en bucle una ejecución real
del clúster previamente grabada. El clúster MPI no puede vivir en un servidor
web (haría falta multiproceso con red entre nodos y bastante CPU), y se ejecuta
en local con Docker.

```
Render  ──►  imagen Docker de 2 etapas
              │
              ├── etapa 1 · node:20-alpine   npm ci + vite build  →  static/dist
              └── etapa 2 · python:3.12-slim Flask + gunicorn sirviendo:
                                              · la SPA de React ya compilada
                                              · la API y el WebSocket
                                              · demo/ejecucion_real.jsonl
```

> **Importante:** `DASHBOARD_VOTACION/static/dist/` **no está en el repositorio**
> (es un artefacto de compilación, está en `.gitignore`). La imagen Docker lo
> genera durante el build. Por eso Render tiene que construir con Docker y no
> con un runtime de Python a secas.

La imagen final pesa **208 MB** y no lleva Node: la etapa de compilación se
descarta y solo se copian los estáticos ya construidos.

---

## Por qué Render y no Fly.io o Vercel

| Plataforma | Veredicto |
|---|---|
| **Render** ✅ | Plan gratuito sin tarjeta, despliega desde GitHub leyendo `render.yaml`, soporta builds Docker multi-etapa y conexiones persistentes |
| Fly.io | Ya no tiene plan gratuito: pide tarjeta incluso para una app mínima. Se incluye [`fly.toml`](../fly.toml) por si migras |
| Vercel / Netlify | Serverless: no mantienen un proceso vivo, y el modo replay lo necesita |
| Railway | Solo 5 $ de crédito de prueba, luego de pago |

El único inconveniente de Render es que **duerme el servicio tras 15 minutos sin
visitas** y despertarlo tarda ~50 s. Eso ya está resuelto: el workflow
[`keep-alive.yml`](../.github/workflows/keep-alive.yml) lo despierta cada 14
minutos desde GitHub Actions, gratis.

---

## Paso 1 · Subir el código a GitHub

Desde la raíz del proyecto:

```bash
git add -A
git commit -m "Moderniza el proyecto: build reparado, tests, CI, Docker y panel React"
git push -u origin modernizacion
```

Después, en GitHub, abre un Pull Request de `modernizacion` a `main-limpio` y
fusiónalo (o empuja directamente a `main-limpio` si lo prefieres).

**Comprueba antes de subir** que estos dos ficheros van dentro, porque sin ellos
la demo arranca vacía:

```bash
git ls-files | grep -E "ejecucion_real.jsonl|frontend/package.json"
```

Debe devolver:

```
DASHBOARD_VOTACION/demo/ejecucion_real.jsonl     ← la grabación (376 KB, 160 eventos)
DASHBOARD_VOTACION/frontend/package.json         ← para que Docker compile la SPA
```

---

## Paso 2 · Crear el servicio en Render

1. Entra en **[render.com](https://render.com)** y regístrate con tu cuenta de GitHub.
2. Pulsa **New +** → **Blueprint**.
3. Selecciona el repositorio **`gcdavidq/CPyD-Project`**.
4. Render detecta [`render.yaml`](../render.yaml) y muestra el servicio
   `votacion-distribuida-dashboard`. Pulsa **Apply**.
5. Espera al primer despliegue. **Tarda entre 6 y 10 minutos**: descarga dos
   imágenes base, instala las dependencias de npm y compila el frontend. Los
   despliegues siguientes son más rápidos porque Render reutiliza capas.

No hay que configurar nada más: el `render.yaml` ya trae el modo replay, la ruta
de la grabación, la velocidad de reproducción y la generación automática de la
`SECRET_KEY`.

Al terminar tendrás una URL del estilo:

```
https://votacion-distribuida-dashboard.onrender.com
```

### Comprobaciones

| URL | Qué debe pasar |
|---|---|
| `/health` | Responde `{"status":"ok","modo":"replay"}` |
| `/panel` | Las cifras empiezan a moverse en pocos segundos |
| `/nodos` | Cuatro fichas de nodo con uso de CPU y lotes |
| `/mapa` | Mapa del Perú con departamentos coloreados |
| `/` | Página de presentación del proyecto |

Si `/panel` carga pero se queda a cero, mira la sección de problemas al final.

---

## Paso 3 · Poner la URL real en el README

El README apunta ahora mismo a una URL de ejemplo. Sustitúyela por la tuya:

```bash
sed -i 's|https://votacion-distribuida-dashboard.onrender.com|https://TU-URL-REAL.onrender.com|' README.md
```

Y también en el botón «Ver el código en GitHub» del panel, si cambias de repo:
está en [`frontend/src/vistas/Sobre.tsx`](../DASHBOARD_VOTACION/frontend/src/vistas/Sobre.tsx),
constante `REPO`.

```bash
git commit -am "Apunta el README a la demo desplegada"
git push
```

---

## Paso 4 · Evitar que la demo se duerma

1. En GitHub: **Settings** → **Secrets and variables** → **Actions** → pestaña **Variables**.
2. **New repository variable**:
   - Nombre: `DEMO_URL`
   - Valor: `https://TU-URL-REAL.onrender.com`
3. Listo. El workflow `keep-alive` hará ping cada 14 minutos.

> Si no defines la variable, el workflow no falla: simplemente avisa y termina.

---

## Paso 5 · Comprobar que la CI pasa

En la pestaña **Actions** del repositorio debería verse el workflow **CI** en
verde, con **cuatro trabajos**:

| Trabajo | Qué verifica |
|---|---|
| **C++ / MPI / OpenMP** | Compila con MPI y ejecuta los 29 tests |
| **Frontend (React + TypeScript)** | `tsc --noEmit` sin errores y `vite build` correcto |
| **Generador de datos + dashboard** | Genera un CSV y comprueba que las 6 rutas responden 200 |
| **Imágenes Docker** | Construye las dos imágenes |

`Generador de datos + dashboard` depende de `Frontend`: reutiliza el build que
aquel sube como artefacto, en vez de compilarlo dos veces.

Cuando esté verde, el badge del README se pondrá en verde solo.

---

## Ejecutar todo en local

### Con Docker (no necesitas Node ni MPI)

```bash
docker compose up --build
```

Levanta dos servicios:

| Servicio | Imagen | Qué hace |
|---|---|---|
| `dashboard` | 208 MB | Sirve el panel en `http://localhost:5000` |
| `cluster` | 633 MB | Genera los datos y ejecuta 1 maestro + 4 esclavos |

La primera vez tarda unos **10-15 minutos**: compila el sistema C++ con MPI,
construye la SPA y genera 120 MB de votos sintéticos. Los CSV quedan en `DATA/`
(montado como volumen), así que las siguientes ejecuciones los reutilizan y
arrancan en segundos.

El clúster es un trabajo por lotes: cuando termina de procesar, su contenedor
sale. El dashboard se queda levantado con los resultados.

### Sin Docker

El panel es una SPA de React, así que **hay que compilarlo antes de servirlo**:

```bash
cd DASHBOARD_VOTACION/frontend
npm install && npm run build     # necesita Node 20+
```

Si te saltas este paso, Flask responde con una página que te lo recuerda en
lugar de fallar con un error críptico.

Para trabajar en el frontend con recarga en caliente:

```bash
# terminal 1 — backend
cd DASHBOARD_VOTACION && python app.py

# terminal 2 — frontend con hot reload en http://localhost:5173
cd DASHBOARD_VOTACION/frontend && npm run dev
```

El servidor de Vite redirige `/api` y `/socket.io` al Flask del puerto 5000, así
que se ve con datos reales del clúster mientras editas.

---

## Regenerar la grabación de la demo

Si cambias el algoritmo y quieres que la demo refleje los nuevos resultados:

```bash
# 1. Genera datos (si no los tienes)
python3 SCRIPTS/generador_votos.py --todas --escala 10 --semilla 42

# 2. Arranca el dashboard EN MODO GRABACIÓN
cd DASHBOARD_VOTACION
MODO_DASHBOARD=live GRABAR_EJECUCION=1 python3 app.py &

# 3. Lanza el clúster (en otra terminal).
#    El intervalo corto es a propósito: con el valor por defecto de 60 s la
#    grabación tendría solo dos o tres fotogramas.
VOTACION_INTERVALO_REPORTE_SEG=3 mpirun --oversubscribe -np 5 ./build/votacion 3

# 4. Al terminar, la nueva grabación está en
#    DASHBOARD_VOTACION/demo/ejecucion_real.jsonl
git commit -am "Actualiza la grabación de la demo"
git push
```

Render redesplegará solo al detectar el push.

**Comprueba que la grabación trae métricas de nodo**, o la vista de clúster
saldrá vacía:

```bash
grep -c '"tipo": "nodo"' DASHBOARD_VOTACION/demo/ejecucion_real.jsonl
```

Debería dar bastante más de 5. Si da exactamente 5, solo se grabó el estado
inicial de los nodos y algo va mal en el envío del maestro.

---

## Solución de problemas

**El build del clúster falla con «No SOURCES given to target»**
El `Dockerfile` del clúster no copia `tests/` ni `benchmarks/`, así que compila
con `-DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF`. Si añades un objetivo nuevo a
`CMakeLists.txt` que dependa de fuentes fuera de `VOTACION/`, ponlo detrás de su
propia opción o cópialas en el Dockerfile. La CI comprueba esta configuración
mínima precisamente para que no se escape.

**El despliegue falla en la etapa de Node**
Mira los logs de Render. Lo más habitual es que `package-lock.json` no esté en
el repositorio: el Dockerfile intenta `npm ci` y cae a `npm install`, pero sin
el lock las versiones pueden no resolverse igual. Comprueba con
`git ls-files | grep package-lock`.

**El panel muestra «El frontend no está compilado»**
Falta `static/dist` dentro de la imagen. Significa que la etapa de build no
produjo nada o que la ruta del `COPY --from` no cuadra. En local se reproduce
con `docker build ./DASHBOARD_VOTACION` y mirando la salida de `vite build`.

**La demo carga pero los números se quedan en cero**
El modo replay no encontró la grabación. Comprueba que
`DASHBOARD_VOTACION/demo/ejecucion_real.jsonl` está en el repositorio y que
`MODO_DASHBOARD=replay`. Con `/api/serie` puedes ver si el servidor tiene
historial.

**La vista de clúster sale vacía pero el resumen funciona**
La grabación no tiene eventos de nodo con datos reales (ver arriba). Regrábala.

**La primera visita tarda casi un minuto**
Es el arranque en frío de Render tras dormirse. Configura `DEMO_URL` (paso 4).

**El ritmo de votos por segundo parece disparado**
En replay el reloj corre acelerado (`REPLAY_VELOCIDAD`). El panel ya divide por
ese factor, que obtiene de `/api/info`. Si ves cifras irreales, comprueba que
ese endpoint devuelve la velocidad correcta.
