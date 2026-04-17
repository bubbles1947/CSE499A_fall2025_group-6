# LocalMind AI — Project Reference for Claude

## What This Project Is

LocalMind AI is a fully private, self-hosted AI chat platform. All inference runs on the local machine — no cloud APIs, no data sent externally. It supports multi-model LLM chat, voice input/output, and RAG (Retrieval-Augmented Generation) over uploaded PDFs.

---

## Stack

| Layer | Technology |
|---|---|
| Mobile / Web | React Native (Expo SDK 54) + React Navigation + Zustand |
| Backend | FastAPI + SQLAlchemy 2 (async) + Pydantic v2 |
| LLM Inference | llama-cpp-python (OpenAI-compatible server) |
| Embeddings | SentenceTransformers all-MiniLM-L6-v2 (embedded, no server) |
| Vector DB | ChromaDB PersistentClient (local file store in `backend/chroma_data/`) |
| Voice STT | OpenAI Whisper (local, `whisper` package) |
| Voice TTS | Coqui TTS (`tts_models/en/ljspeech/tacotron2-DDC`) |
| Database | PostgreSQL 16 (Docker) |
| Auth | JWT (python-jose) + bcrypt 4.1.3 |

---

## Credentials

| Service | Value |
|---|---|
| Test account | admin@test.com / admin12345 |
| PostgreSQL | host: localhost:5432, db: localmind, user: postgres |
| JWT secret | see `backend/.env` → `JWT_SECRET_KEY` |

---

## How to Start Everything

### 1. Start infrastructure (Docker)
```bash
cd backend
docker compose up -d
# PostgreSQL on :5432, ChromaDB on :8005
```

### 2. Start backend
```bash
cd backend
source venv/Scripts/activate        # Windows (Git Bash)
# source venv/bin/activate           # Mac/Linux
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start LLM model server
```bash
# From project root — runs DeepSeek R1 1.5B on port 8003
start_models.bat                    # Windows
# Or manually:
cd backend
source venv/Scripts/activate
python -m llama_cpp.server \
  --model models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf \
  --host 0.0.0.0 --port 8003 \
  --n_ctx 2048 --n_threads 4
```

### 4. Start mobile/web app
```bash
cd mobile
npx expo start
# Press w for web (http://localhost:8081)
# Scan QR for Expo Go on phone
```

---

## Model Servers

| Model | Port | File |
|---|---|---|
| Llama 3.1 8B | 8001 | models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf |
| Mistral 7B | 8002 | models/mistral-7b-instruct-v0.2.Q4_K_M.gguf |
| DeepSeek R1 1.5B | 8003 | models/deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf |

Currently only DeepSeek is downloaded and working. Model files live in `backend/models/`.

---

## API Endpoints

```
POST  /auth/register          Register new user
POST  /auth/login             Login → JWT token

GET   /chat/models            List models + live status
POST  /chat/send              Stream LLM response (SSE)
GET   /chat/conversations     List all conversations
GET   /chat/conversations/:id Get conversation with messages
DELETE /chat/conversations/:id Delete conversation

POST  /rag/upload             Upload PDF → process → ChromaDB
GET   /rag/documents          List uploaded documents
DELETE /rag/documents/:id     Delete document + embeddings
POST  /rag/chat               RAG chat (retrieve context + LLM answer)

POST  /voice/transcribe       Whisper STT (multipart audio)
POST  /voice/synthesize       Coqui TTS (returns WAV)
GET   /voice/synthesize_get   TTS via GET (for FileSystem.downloadAsync)
```

---

## Folder Structure

```
localmind-ai/
├── backend/
│   ├── app/
│   │   ├── main.py               FastAPI app entry, CORS, routers
│   │   ├── config.py             All settings (env vars + defaults)
│   │   ├── database.py           Async SQLAlchemy session
│   │   ├── auth.py               JWT encode/decode, get_current_user
│   │   ├── models/               SQLAlchemy ORM models
│   │   │   ├── user.py
│   │   │   ├── conversation.py
│   │   │   ├── message.py
│   │   │   └── document.py
│   │   ├── schemas/              Pydantic request/response schemas
│   │   │   ├── auth.py
│   │   │   ├── chat.py
│   │   │   ├── rag.py
│   │   │   └── voice.py
│   │   ├── routers/
│   │   │   ├── auth.py           Register, login
│   │   │   ├── chat.py           SSE streaming, model health pings
│   │   │   ├── rag.py            PDF upload, RAG chat
│   │   │   └── voice.py          Whisper STT, Coqui TTS
│   │   └── services/
│   │       └── rag.py            PDF→chunks→embeddings→ChromaDB
│   ├── alembic/                  DB migrations
│   ├── models/                   GGUF model files (not in git)
│   ├── uploads/                  Uploaded PDFs
│   ├── chroma_data/              ChromaDB persistent storage
│   ├── venv/                     Python virtual environment
│   ├── .env                      Environment variables
│   ├── requirements.txt
│   └── docker-compose.yml
├── mobile/
│   ├── src/
│   │   ├── navigation/
│   │   │   └── AppNavigator.tsx  Root stack + tab navigator
│   │   ├── screens/
│   │   │   ├── LoginScreen.tsx
│   │   │   ├── RegisterScreen.tsx
│   │   │   ├── HomeScreen.tsx    Conversation list + model status
│   │   │   ├── ChatScreen.tsx    SSE streaming chat + RAG mode
│   │   │   └── DocumentScreen.tsx PDF upload + management
│   │   ├── components/
│   │   │   ├── MessageBubble.tsx  Markdown rendering + RAG sources
│   │   │   ├── TypingIndicator.tsx Animated 3-dot bounce
│   │   │   ├── ChatHistoryDrawer.tsx Bottom sheet history
│   │   │   ├── MicButton.tsx     4-state STT button
│   │   │   ├── SpeakerButton.tsx TTS toggle
│   │   │   └── ModelPicker.tsx   Model selector with status
│   │   ├── store/
│   │   │   ├── useAuthStore.ts   Auth state + JWT persistence
│   │   │   └── useChatStore.ts   Chat, RAG, voice, model state
│   │   └── services/
│   │       ├── api.ts            Axios instance (platform-split base URL)
│   │       └── voice.ts          Whisper STT + Coqui TTS (web + native)
│   └── package.json
├── start_models.bat              Launch DeepSeek on port 8003
├── CLAUDE.md                     This file
└── README.md                     Setup guide
```

---

## Key Implementation Notes

### SSE Streaming
Backend: `StreamingResponse` with `text/event-stream`. Mobile: native `fetch()` with `response.body.getReader()` (axios doesn't stream on RN). Conversation ID returned in `X-Conversation-Id` header.

### DeepSeek `<think>` tags
DeepSeek R1 emits `<think>...</think>` chain-of-thought blocks. These are stripped via regex in:
- `MessageBubble.tsx` before rendering
- `ChatScreen.tsx` streaming bubble
- `useChatStore.ts` before TTS playback

### Voice (platform split)
- **Web**: `navigator.mediaDevices.getUserMedia` + `MediaRecorder` + `fetch()` for upload
- **Native**: `expo-av Audio.Recording` + `axios` multipart for upload
- **TTS web**: `fetch` → Blob → `URL.createObjectURL` → `new Audio()`
- **TTS native**: `FileSystem.downloadAsync` → `expo-av Sound`

### RAG Pipeline
1. Upload PDF → `PyPDFLoader` → `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
2. `SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")` generates embeddings
3. Store in `ChromaDB.PersistentClient` (local, no server needed)
4. Query: embed question → retrieve top-5 chunks → build system prompt → call LLM

### Model Health
`GET /chat/models` pings all 3 servers concurrently via `asyncio.gather()` with 3s timeout. Pre-flight ping before each `sendMessage` prevents orphaned DB records when model is offline.

### Database
PostgreSQL via async SQLAlchemy. Migrations managed with Alembic. Tables: `users`, `conversations`, `messages`, `documents`.

### bcrypt Pin
`bcrypt==4.1.3` pinned in requirements.txt. passlib 1.7.4 is incompatible with bcrypt 5.x.

---

## What Is Built and Working

- [x] JWT auth (register, login, persistent sessions)
- [x] Multi-model LLM chat with real-time SSE streaming
- [x] DeepSeek R1 1.5B running locally (port 8003)
- [x] Model health monitoring (concurrent pings, offline warnings)
- [x] Voice input (Whisper STT) — web + native
- [x] Voice output (Coqui TTS) — web + native  
- [x] RAG: PDF upload → embeddings → chat with document
- [x] Chat history (persistent, per-user)
- [x] Markdown rendering in chat bubbles
- [x] `<think>` tag stripping (DeepSeek chain-of-thought)
- [x] Web Enter key to send; Shift+Enter for newline
- [x] Skeleton loaders, empty states, loading spinners
