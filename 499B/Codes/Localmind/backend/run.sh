#!/usr/bin/env bash
# =============================================================================
# LocalMind AI — Backend Startup Script
# Starts PostgreSQL + ChromaDB via Docker, runs migrations, then FastAPI.
# Run from: localmind-ai/backend/
# =============================================================================

set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Colour

info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warning() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── Prerequisite checks ───────────────────────────────────────────
command -v docker  >/dev/null 2>&1 || error "Docker is not installed or not in PATH."
command -v python  >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || error "Python is not installed."
PYTHON=$(command -v python3 2>/dev/null || command -v python)

[ -f ".env" ] || error ".env file not found. Copy .env.example to .env and fill in your values."

# ── Step 1: Start Docker services ─────────────────────────────────
info "Starting PostgreSQL and ChromaDB via Docker Compose..."
docker compose -f ../docker-compose.yml up -d

# Wait for PostgreSQL to be healthy
info "Waiting for PostgreSQL to be ready..."
RETRIES=30
until docker exec localmind-postgres pg_isready -U postgres -d localmind >/dev/null 2>&1; do
    RETRIES=$((RETRIES - 1))
    if [ "$RETRIES" -le 0 ]; then
        error "PostgreSQL did not become ready in time."
    fi
    sleep 2
done
info "PostgreSQL is ready."

# ── Step 2: Run Alembic migrations ────────────────────────────────
info "Running database migrations..."
$PYTHON -m alembic upgrade head
info "Migrations complete."

# ── Step 3: Start FastAPI server ──────────────────────────────────
info "Starting FastAPI server on http://0.0.0.0:8000 ..."
info "Swagger UI available at: http://localhost:8000/docs"
info "Press Ctrl+C to stop."
echo ""

$PYTHON -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
