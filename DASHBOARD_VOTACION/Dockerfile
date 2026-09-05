# =============================================================================
#  Imagen del dashboard web
#
#  Sirve tanto para desarrollo local (docker compose) como para el despliegue
#  publico en Render. El modo lo decide la variable MODO_DASHBOARD.
#
#  El frontend es una SPA de React que se compila en la primera etapa; la
#  imagen final no lleva Node, solo los estaticos ya construidos.
# =============================================================================

# ---- Etapa 1: compilar el frontend ------------------------------------------
FROM node:20-alpine AS frontend

WORKDIR /build

# Las dependencias se copian aparte para que Docker reutilice la cache siempre
# que package.json no cambie.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund 2>/dev/null || npm install --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ---- Etapa 2: servidor -------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY demo/ ./demo/
COPY static/data/ ./static/data/
COPY static/img/ ./static/img/

# Los estaticos compilados vienen de la etapa anterior. Vite escribe en
# `../static/dist` respecto de frontend/, que dentro del contenedor (WORKDIR
# /build) resuelve a /static/dist.
COPY --from=frontend /static/dist ./static/dist

# Usuario sin privilegios: nada de este servicio necesita root.
RUN useradd --create-home dashboard && chown -R dashboard:dashboard /app
USER dashboard

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=3)"

# Un solo worker: el estado de las estadisticas vive en memoria del proceso y
# las conexiones de Socket.IO son persistentes, asi que no se puede escalar
# horizontalmente sin un backend de mensajes (Redis) por delante. Los hilos
# sirven para atender varias conexiones de long-polling a la vez.
CMD ["gunicorn", \
     "--worker-class", "gthread", \
     "--threads", "16", \
     "--workers", "1", \
     "--bind", "0.0.0.0:5000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "app:app"]
