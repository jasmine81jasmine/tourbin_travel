# --- Tourbin backend image -------------------------------------------------
# Mirrors the exact folder layout the app expects when run locally:
#   /app/index.html
#   /app/backend/src/main.py   (uvicorn is started with cwd = /app/backend)
FROM python:3.11-slim

# Runtime deps: curl is used by the container HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache this layer between builds.
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt

# Now copy the actual app code.
COPY index.html ./index.html
COPY backend ./backend

WORKDIR /app/backend

EXPOSE 8000

# Docker (and docker compose) will mark the container unhealthy if this fails,
# which is what the deploy pipeline checks before calling a deploy "successful".
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl -fs http://localhost:8000/health | grep -q '"status":"healthy"' || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
