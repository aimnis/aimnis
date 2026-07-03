# Aimnis web service: dashboard + HTTP gateway (/v1) in one process.
# Build context is the `server/` directory (set the Railway service root to /server).
FROM python:3.12-slim

# libgomp1 is required by onnxruntime (fastembed's backend for local embeddings).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app/server
COPY . /app/server

# Editable install so `aimnis.migrate` resolves migrations/ at ../migrations
# (parent.parent of the package) exactly as in dev. Include the api + mcp extras.
RUN uv pip install --system --no-cache -e ".[api,mcp]"

ENV PYTHONUNBUFFERED=1
# fastembed downloads its ~130MB ONNX model on the first cache-miss embed; cache it
# under /app so a warm container reuses it.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed_cache

# Railway injects $PORT. Run migrations first (idempotent), then serve on 0.0.0.0.
CMD ["sh", "-c", "python -m aimnis.migrate && uvicorn aimnis.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
