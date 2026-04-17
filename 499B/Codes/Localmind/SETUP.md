# LocalMind AI — Setup Guide

Complete step-by-step instructions to get the backend running from scratch.

---

## Prerequisites

Before you begin, make sure you have:

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.11+ | Backend runtime |
| Docker Desktop | Latest | PostgreSQL + ChromaDB |
| Git | Any | Source control |
| VS Code | Any | Optional — for `.http` test file |
| VS Code "REST Client" extension | `humao.rest-client` | Running `.http` tests |

---

## Step 1 — Clone / Open the Project

Open a terminal (Git Bash, PowerShell, or WSL) and navigate to the project root:

```bash
cd C:/Users/MSI/Desktop/localmind-ai
```

---

## Step 2 — Create a Python Virtual Environment

Always work inside a virtual environment to isolate dependencies.

```bash
cd backend

# Create the venv
python -m venv venv

# Activate it — Windows (Git Bash / WSL)
source venv/Scripts/activate

# Activate it — Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate it — macOS / Linux
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

---

## Step 3 — Install Python Dependencies

```bash
# Make sure venv is active, then:
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on heavy packages:**
> - `openai-whisper` downloads model weights on first use (~150 MB for `base`)
> - `TTS` (Coqui) downloads its model on first use (~200 MB)
> - `llama-cpp-python` may need CUDA build flags if using GPU — see below

**Optional: GPU-accelerated llama-cpp-python (NVIDIA CUDA)**
```bash
pip uninstall llama-cpp-python -y
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

---

## Step 4 — Configure Environment Variables

```bash
# From inside backend/
cp .env.example .env
```

Open `.env` in your editor and fill in / verify:

```env
# Change this to a long random string — never commit the real secret!
JWT_SECRET_KEY=my-super-secret-random-key-here

# These match the docker-compose.yml defaults — change only if you customised them
DATABASE_URL=postgresql+asyncpg://localmind:localmind_secret@localhost:5432/localmind_db
CHROMA_HOST=localhost
CHROMA_PORT=8005

# LLM servers — start these separately with llama-cpp-python (see Step 6)
LLAMA3_SERVER_URL=http://localhost:8001/v1
MISTRAL_SERVER_URL=http://localhost:8002/v1
DEEPSEEK_SERVER_URL=http://localhost:8003/v1
EMBEDDINGS_SERVER_URL=http://localhost:8004/v1
```

---

## Step 5 — Start Docker Services (PostgreSQL + ChromaDB)

```bash
# From localmind-ai/ (project root)
cd ..
docker compose up -d

# Verify both containers are healthy
docker compose ps
```

Expected output:
```
NAME                   STATUS
localmind-postgres     Up (healthy)
localmind-chromadb     Up
```

To stop them later:
```bash
docker compose down          # stop only
docker compose down -v       # stop + delete all data
```

---

## Step 6 — Start LLM Servers (llama-cpp-python)

Each model runs as its own HTTP server. Open **separate terminals** for each.

> **Place your GGUF model files in `backend/models/`.**
> Download from HuggingFace — look for Q4_K_M quantization as a good speed/quality balance.

```bash
# Terminal 1 — Llama 3.1 8B (port 8001)
cd backend && source venv/Scripts/activate
python -m llama_cpp.server \
  --model models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8001 \
  --n_ctx 8192 --chat_format llama-3

# Terminal 2 — Mistral 7B (port 8002)
python -m llama_cpp.server \
  --model models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8002 \
  --n_ctx 8192 --chat_format mistral-instruct

# Terminal 3 — DeepSeek R1 (port 8003)
python -m llama_cpp.server \
  --model models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8003 \
  --n_ctx 8192

# Terminal 4 — nomic-embed-text Embeddings (port 8004)
python -m llama_cpp.server \
  --model models/nomic-embed-text-v1.5.Q4_K_M.gguf \
  --host 0.0.0.0 --port 8004 \
  --n_ctx 2048 \
  --embedding True
```

> You don't need all 4 running at once — start only the models you want to test.
> The `/chat/models` endpoint reports `"status": "online"` or `"offline"` for each.

---

## Step 7 — Run Database Migrations

With Docker running and venv active:

```bash
# From backend/
cd backend   # if not already there
alembic upgrade head
```

Expected output:
```
INFO  [alembic.runtime.migration] Running upgrade  -> 001, initial tables
```

To check current migration state:
```bash
alembic current
```

---

## Step 8 — Start the FastAPI Server

**Option A — Quick start script (does steps 5 → 7 → 8 automatically):**
```bash
cd backend
bash run.sh
```

**Option B — Manual (more control):**
```bash
cd backend
source venv/Scripts/activate   # make sure venv is active

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at **http://localhost:8000**

| URL | Description |
|-----|-------------|
| http://localhost:8000/docs | Swagger UI — interactive API explorer |
| http://localhost:8000/redoc | ReDoc — clean API reference |
| http://localhost:8000/health | Health check endpoint |

---

## Step 9 — Test the API with test_api.http

### Setup
1. Install the **REST Client** extension in VS Code: `humao.rest-client`
2. Open `backend/test_api.http`

### Workflow
1. **Click "Send Request"** above `### 1. Register a new user`
2. **Click "Send Request"** above `### 2. Login`
3. In the response panel, copy the `access_token` value
4. Paste it into the `@token = ` variable at the top of the file
5. Run any other request — they all use `Bearer {{token}}` automatically

### Testing the streaming chat
Click "Send Request" on `### 6. Send a message (streaming SSE)`.
The response panel streams `data: {...}` lines in real time.
Check the response headers for **`X-Conversation-Id`** — copy this UUID for follow-up messages.

### File uploads (voice/rag) via curl
```bash
# Transcribe audio
curl -X POST http://localhost:8000/voice/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "audio=@/path/to/audio.wav"

# Upload a PDF for RAG
curl -X POST http://localhost:8000/rag/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@/path/to/document.pdf"
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `alembic: command not found` | Make sure venv is active: `source venv/Scripts/activate` |
| `connection refused` on DB | Docker not running — `docker compose up -d` |
| `DATABASE_URL is not set` | `.env` file missing — `cp .env.example .env` |
| LLM model shows `offline` | That llama-cpp-python server isn't started yet |
| `No module named app` | Run uvicorn from inside `backend/`, not the project root |
| Port already in use | Kill the old process: `lsof -i :8000` then `kill <PID>` |
| Whisper slow on first call | It's downloading model weights — subsequent calls are fast |

---

## Directory Quick Reference

```
localmind-ai/
├── backend/
│   ├── alembic/            # Migration scripts
│   ├── alembic.ini         # Alembic config
│   ├── app/
│   │   ├── main.py         # FastAPI entry point
│   │   ├── config.py       # Settings (reads .env)
│   │   ├── database.py     # Async SQLAlchemy engine
│   │   ├── auth.py         # JWT + password helpers
│   │   ├── models/         # SQLAlchemy ORM models
│   │   ├── schemas/        # Pydantic request/response schemas
│   │   ├── routers/        # API route handlers
│   │   └── services/       # Business logic (RAG, voice, LLM)
│   ├── models/             # GGUF model files go here
│   ├── uploads/            # User-uploaded PDFs stored here
│   ├── .env                # Your local env vars (never commit!)
│   ├── .env.example        # Template — safe to commit
│   ├── requirements.txt    # Python dependencies
│   ├── run.sh              # One-command startup script
│   └── test_api.http       # REST Client test requests
├── docker-compose.yml      # PostgreSQL + ChromaDB
└── SETUP.md                # This file
```
