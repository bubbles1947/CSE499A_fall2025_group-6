# LocalMind AI

A fully private, self-hosted AI chat platform. All inference runs on your own hardware — no cloud APIs, no data leaves your machine.

**Features:** Multi-model LLM chat · Real-time streaming · Voice input (Whisper STT) · Voice output (Coqui TTS) · RAG over PDFs · Chat history · Web + Mobile

---

## Requirements

- Windows 10/11 (tested), macOS, or Linux
- Python 3.11+
- Node.js 18+
- Docker Desktop
- 8 GB RAM minimum (16 GB recommended for 7B models)
- ffmpeg (for voice transcription)

---

## Quick Start

### 1. Clone and set up backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate      # Mac/Linux

# Install dependencies
pip install -r requirements.txt
pip install "llama-cpp-python[server]"
```

### 2. Start the database

```bash
docker compose up -d
```

This starts PostgreSQL (port 5432) and ChromaDB (port 8005).

### 3. Run database migrations

```bash
source venv/Scripts/activate
alembic upgrade head
```

### 4. Create a `.env` file

```bash
cp .env.example .env   # or create manually
```

Minimum required content:
```
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/localmind
JWT_SECRET_KEY=change-me-to-a-random-secret-key
```

### 5. Download a model

```bash
mkdir -p models
# Download DeepSeek R1 1.5B (smallest, ~1GB, runs on CPU)
curl -L -o models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf \
  https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf
```

For larger models (better quality, need more RAM):
```bash
# Mistral 7B (~4.4GB, needs 8GB RAM)
# Llama 3.1 8B (~4.9GB, needs 10GB RAM)
```

### 6. Start the model server

```bash
# Windows — double-click or run:
start_models.bat

# Mac/Linux:
source venv/bin/activate
python -m llama_cpp.server \
  --model models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf \
  --host 0.0.0.0 --port 8003 \
  --n_ctx 2048 --n_threads 4
```

### 7. Start the backend

```bash
source venv/Scripts/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at http://localhost:8000. API docs at http://localhost:8000/docs.

### 8. Start the mobile/web app

```bash
cd mobile
npm install
npx expo start
```

- Press **w** to open in browser at http://localhost:8081
- Scan the **QR code** with Expo Go app on your phone
- Press **a** for Android emulator, **i** for iOS simulator

### 9. Install ffmpeg (for voice transcription)

**Windows (winget):**
```bash
winget install Gyan.FFmpeg
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

---

## Usage

### Chat
1. Register an account or log in
2. Tap **New Chat** on the home screen
3. Select a model from the dropdown (green dot = online)
4. Type a message and press Send (or Enter on web)
5. Use the **mic button** to speak your message
6. Use the **speaker button** to have AI responses read aloud

### RAG (Chat with a PDF)
1. Go to the **Documents** tab
2. Tap **Upload PDF** and select a file
3. Wait for status to show **ready** (embedding takes ~10–30 seconds)
4. Tap the chat bubble icon on the document
5. Ask questions about the document's content

### Chat History
- Tap the **clock icon** in the chat header to open history
- Select any past conversation to resume it
- Long-press on the home screen to delete a conversation

---

## Model Configuration

Edit `backend/app/config.py` to change server URLs:

```python
LLAMA3_SERVER_URL   = "http://localhost:8001/v1"
MISTRAL_SERVER_URL  = "http://localhost:8002/v1"
DEEPSEEK_SERVER_URL = "http://localhost:8003/v1"
```

Each model needs its own llama-cpp-python server instance on a different port.

---

## Mobile on Physical Device

Update the API base URL in `mobile/src/services/api.ts`:

```typescript
const API_BASE_URL =
  Platform.OS === "web"
    ? "http://localhost:8000"
    : "http://YOUR_PC_IP:8000";   // ← replace with your local IP
```

Find your IP with `ipconfig` (Windows) or `ifconfig` (Mac/Linux).

---

## Architecture

```
[React Native App]
       ↓ REST + SSE
[FastAPI Backend :8000]
       ↓                    ↓                ↓
[llama-cpp-python   [PostgreSQL      [ChromaDB
 LLM servers        :5432]           local files]
 :8001/8002/8003]
       ↓
[Whisper STT + Coqui TTS (in-process)]
```

---

## Troubleshooting

**Model shows "offline"** — Make sure the llama-cpp-python server is running on the correct port. Run `start_models.bat`.

**Voice transcription fails** — Install ffmpeg and ensure it's in PATH. On Windows, run `winget install Gyan.FFmpeg`.

**"Connection refused" on mobile** — Use your PC's local IP address (not `localhost`) in `api.ts`.

**PDF upload fails** — Check that the backend `uploads/` directory exists and is writable.

**Slow responses** — DeepSeek R1 1.5B runs on CPU. Expect ~20–100 tokens/sec depending on hardware. Use `--n_threads` equal to your CPU core count.
