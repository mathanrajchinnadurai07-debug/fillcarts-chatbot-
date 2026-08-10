#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  Curify AI Advisor — One-Command Local Deploy Script         ║
# ║  Usage: bash deploy.sh [mode]                                ║
# ║  Modes:                                                      ║
# ║    dev    (default) — local uvicorn, no Docker               ║
# ║    docker           — build + run via docker compose         ║
# ║    train            — train model then start dev server      ║
# ║    test             — run full test suite                     ║
# ║    clean            — remove all generated artefacts         ║
# ╚══════════════════════════════════════════════════════════════╝

set -euo pipefail

# ─── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

# ─── Helpers ──────────────────────────────────────────────────────────────────
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

banner() {
  echo -e "${BLUE}"
  echo "╔══════════════════════════════════════════════════════╗"
  echo "║       Curify AI Advisor — Local Deploy Script        ║"
  echo "╚══════════════════════════════════════════════════════╝"
  echo -e "${NC}"
}

# ─── Config ───────────────────────────────────────────────────────────────────
MODE="${1:-dev}"
PYTHON="${PYTHON:-python}"
VENV_DIR=".venv"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# ─── Pre-flight checks ────────────────────────────────────────────────────────
check_python() {
  if ! command -v "$PYTHON" &>/dev/null; then
    error "Python not found. Install Python 3.11+ and try again."
  fi
  PY_VER=$("$PYTHON" --version 2>&1 | awk '{print $2}')
  info "Python version: $PY_VER"
}

check_env_file() {
  if [[ ! -f ".env" ]]; then
    warn ".env file not found. Copying from .env.example..."
    cp .env.example .env
    warn "Please edit .env and set your GEMINI_API_KEY before continuing."
  fi
}

setup_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment in $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
  fi

  # Activate venv
  if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # Windows
    source "$VENV_DIR/Scripts/activate"
  elif [[ -f "$VENV_DIR/bin/activate" ]]; then
    # Unix/macOS
    source "$VENV_DIR/bin/activate"
  fi

  success "Virtual environment active."
}

install_deps() {
  info "Installing dependencies from requirements.txt..."
  pip install --upgrade pip --quiet
  pip install -r requirements.txt --quiet
  success "Dependencies installed."
}

create_dirs() {
  mkdir -p models/saved data/processed chroma_db logs
  success "Required directories created."
}

# ─── Modes ────────────────────────────────────────────────────────────────────

run_dev() {
  info "Starting Curify AI Advisor in DEV mode on http://${HOST}:${PORT}..."
  info "API docs available at http://localhost:${PORT}/docs"
  echo ""
  PYTHONPATH=. uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --log-level info
}

run_train() {
  info "Training the intent classifier and indexing ChromaDB..."
  PYTHONPATH=. python models/train.py \
    --csv ./data/raw/sample_training_data.csv \
    --reset-chroma
  success "Training complete. Model saved to models/saved/"
  echo ""
  info "Starting server..."
  run_dev
}

run_docker() {
  if ! command -v docker &>/dev/null; then
    error "Docker not found. Install Docker Desktop and try again."
  fi
  if ! command -v docker compose &>/dev/null 2>&1; then
    error "Docker Compose not found. Update Docker Desktop to get 'docker compose'."
  fi

  info "Building Docker image..."
  docker compose build

  info "Starting services..."
  docker compose up -d

  info "Waiting for health check..."
  sleep 10

  HEALTH=$(curl -s "http://localhost:${PORT}/health" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "unreachable")
  if [[ "$HEALTH" == "healthy" ]]; then
    success "App is healthy at http://localhost:${PORT}"
    success "API docs: http://localhost:${PORT}/docs"
  else
    warn "Health check returned: $HEALTH (app may still be starting)"
    info "Check logs with: docker compose logs -f app"
  fi
}

run_tests() {
  info "Running full test suite..."
  PYTHONPATH=. python -m pytest tests/ \
    -v \
    --tb=short \
    --timeout=120 \
    -p no:warnings

  success "All tests passed!"
}

run_clean() {
  warn "Removing generated artefacts (models, chroma_db, db file, logs)..."
  rm -rf models/saved chroma_db curify_chat.db logs __pycache__
  find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  find . -name "*.pyc" -delete 2>/dev/null || true
  success "Clean complete."
}

run_preprocess() {
  info "Running data preprocessing..."
  PYTHONPATH=. python scripts/preprocess_data.py \
    --input ./data/raw/sample_training_data.csv \
    --output ./data/processed/clean_training_data.csv
  success "Preprocessing complete."
}

# ─── Entry Point ──────────────────────────────────────────────────────────────
banner
check_python
check_env_file

case "$MODE" in
  dev)
    setup_venv
    install_deps
    create_dirs
    run_dev
    ;;
  train)
    setup_venv
    install_deps
    create_dirs
    run_train
    ;;
  docker)
    run_docker
    ;;
  test)
    setup_venv
    install_deps
    run_tests
    ;;
  preprocess)
    setup_venv
    install_deps
    run_preprocess
    ;;
  clean)
    run_clean
    ;;
  *)
    echo "Usage: bash deploy.sh [dev|train|docker|test|preprocess|clean]"
    echo ""
    echo "  dev         — Start local dev server with hot-reload (default)"
    echo "  train       — Train model + start dev server"
    echo "  docker      — Build and run via Docker Compose"
    echo "  test        — Run pytest test suite"
    echo "  preprocess  — Preprocess raw training CSV"
    echo "  clean       — Remove generated artefacts"
    exit 1
    ;;
esac
