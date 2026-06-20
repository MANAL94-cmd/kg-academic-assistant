# ── Stage 1: build the React frontend ────────────────────────────────────────
FROM node:20-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build           # emits /app/frontend/dist

# ── Stage 2: Python backend (serves the built frontend + the API) ─────────────
FROM python:3.11-slim

# libgomp1 is required by faiss / torch at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache the HuggingFace model inside the image so first boot doesn't download it.
ENV HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PORT=7860

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Pre-download the sentence-transformer at build time (baked into the image).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY backend/ ./backend/
COPY --from=frontend /app/frontend/dist ./frontend/dist

# HuggingFace Spaces (and the rest) reach the app on 7860.
EXPOSE 7860

CMD ["gunicorn", "--chdir", "backend", "app:app", \
     "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1"]
