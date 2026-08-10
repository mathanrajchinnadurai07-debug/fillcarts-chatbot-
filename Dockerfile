# ╔══════════════════════════════════════════════════════════════╗
# ║  Curify AI Advisor — Multi-stage Production Dockerfile       ║
# ║  Stage 1: builder  — install deps in isolated venv           ║
# ║  Stage 2: runtime  — lean final image (~500 MB vs ~2 GB)     ║
# ╚══════════════════════════════════════════════════════════════╝

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip first for faster resolution
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install requirements
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model into the image
# so containers start without downloading on first request
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"


# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Curify AI Team <support@curify.ai>"
LABEL description="Curify AI Advisor — ML Chatbot with RAG"
LABEL version="1.0.0"

# Runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd --gid 1001 curify && \
    useradd --uid 1001 --gid curify --shell /bin/bash --create-home curify

# Copy virtualenv from builder stage
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy HuggingFace model cache from builder
COPY --from=builder /root/.cache/huggingface /home/curify/.cache/huggingface

# Set working directory
WORKDIR /app

# Copy project files (ordered by change frequency for cache efficiency)
COPY requirements.txt .env.example ./
COPY app/ ./app/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create directories for persistent volumes
RUN mkdir -p chroma_db logs data/processed models/saved && \
    chown -R curify:curify /app

# Switch to non-root user
USER curify

# Environment defaults (override via docker-compose or K8s secrets)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    PORT=8000 \
    ENVIRONMENT=production \
    DEBUG=false \
    CHROMA_PERSIST_DIR=/app/chroma_db

EXPOSE 8000

# Health check — polls /health every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default: run the FastAPI app via uvicorn
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
