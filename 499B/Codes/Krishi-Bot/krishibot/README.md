# KrishiBot — AI Agriculture Assistant

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat&logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=flat&logo=next.js&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-FF6B35?style=flat)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat)

KrishiBot is an AI-powered agricultural assistant designed for smallholder farmers in South Asia. It provides crop disease detection from photos, conversational farming advice in English and Bangla, and structured irrigation, fertilizer, and pest-control guidance — all running entirely on local hardware with no cloud dependency.

Built as a Final Year Project, KrishiBot demonstrates how open-source large language models (via Ollama) can be applied to real-world agricultural challenges without subscription fees, internet connectivity requirements, or privacy concerns.

---

## Screenshots

> _Add screenshots of the Chat, Analyze, and Advisory pages here._

---

## Features

- **Conversational Chat** — Ask farming questions in English or Bangla; get expert, contextual answers
- **Crop Disease Detection** — Upload a leaf/crop photo; the vision model identifies diseases and recommends treatment
- **Crop Advisory** — Topic-based guides (irrigation, fertilizer, pest control, disease prevention, harvest) for 5 crops
- **Local AI** — Runs fully offline after initial model download; no API keys or cloud services
- **Session History** — Chat sessions are persisted in SQLite; history survives page refresh
- **Bilingual** — Full Bangla system-prompt support; toggle language in the chat interface
- **Responsive UI** — Works on mobile, tablet, and desktop

---

## Tech Stack

| Layer     | Technology                                   |
|-----------|----------------------------------------------|
| Frontend  | Next.js 14, React 18, TypeScript, Tailwind CSS |
| Backend   | FastAPI 0.109, Python 3.11, SQLAlchemy 2, Pydantic v2 |
| AI Engine | Ollama — `qwen2.5:3b` (text), `qwen2.5vl:3b` (vision) |
| Database  | SQLite (default) — swappable to PostgreSQL via env var |
| HTTP      | httpx (async), SSE streaming, multipart uploads |

---

## Prerequisites

Install these before proceeding:

| Tool         | Version    | Link                                     |
|--------------|------------|------------------------------------------|
| Python       | 3.11+      | https://python.org                       |
| Node.js      | 18+        | https://nodejs.org                       |
| Ollama       | Latest     | https://ollama.com                       |
| Git          | Any        | https://git-scm.com                      |

Hardware minimum: 8 GB RAM (16 GB recommended), 10 GB free disk space.  
A GPU with 4 GB VRAM significantly speeds up inference but is not required.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/krishibot.git
cd krishibot
```

### 2. Backend setup

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create your environment file
cp .env.example .env
# Edit .env if needed (defaults work out of the box for local Ollama)
```

### 3. Frontend setup

```bash
cd ../frontend

# Install dependencies
npm install

# Create environment file
echo 'NEXT_PUBLIC_API_URL=http://localhost:8000' > .env.local
```

### 4. Pull Ollama models

```bash
# Text model — handles chat and advisory
ollama pull qwen2.5:3b

# Vision model — handles image disease detection
ollama pull qwen2.5vl:3b
```

> Model sizes: `qwen2.5:3b` ≈ 2 GB, `qwen2.5vl:3b` ≈ 2.3 GB.

---

## Running the Project

You need **three terminals** running simultaneously.

### Terminal 1 — Ollama inference server

```bash
ollama serve
```

Ollama listens on `http://localhost:11434` by default.

### Terminal 2 — FastAPI backend

```bash
cd krishibot/backend
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API available at: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### Terminal 3 — Next.js frontend

```bash
cd krishibot/frontend
npm run dev
```

App available at: `http://localhost:3000`

---

## API Endpoints

| Method | Endpoint                          | Description                        |
|--------|-----------------------------------|------------------------------------|
| GET    | `/`                               | Root health check                  |
| GET    | `/health`                         | API + Ollama connectivity status   |
| POST   | `/api/chat/message`               | Send message, stream SSE reply     |
| GET    | `/api/chat/history/{session_id}`  | Retrieve session message history   |
| DELETE | `/api/chat/session/{session_id}`  | Clear a chat session               |
| POST   | `/api/analyze/image`              | Upload image, get disease report   |
| GET    | `/api/advisory/crops`             | List supported crops               |
| GET    | `/api/advisory/topics`            | List advisory topics               |
| POST   | `/api/advisory/query`             | Generate crop+topic advisory       |

Full interactive documentation is available at `http://localhost:8000/docs` when the backend is running.

---

## Project Structure

```
krishibot/
├── frontend/                   # Next.js 14 application
│   ├── app/
│   │   ├── page.tsx            # Homepage
│   │   ├── chat/page.tsx       # Chat interface
│   │   ├── analyze/page.tsx    # Image disease detection
│   │   ├── advisory/page.tsx   # Crop advisory
│   │   ├── about/page.tsx      # About & docs
│   │   └── layout.tsx          # Root layout (Navbar + Footer)
│   ├── components/
│   │   ├── Navbar.tsx
│   │   ├── Footer.tsx
│   │   ├── ChatInterface.tsx   # Full chat UI with SSE streaming
│   │   ├── MessageBubble.tsx   # Individual chat message
│   │   ├── ImageUploader.tsx   # Drag-and-drop image upload
│   │   ├── DiseaseReportCard.tsx
│   │   └── AdvisoryCard.tsx
│   └── lib/
│       └── api.ts              # All API client functions + TypeScript types
│
└── backend/                    # FastAPI application
    ├── main.py                 # App entry point, lifespan, CORS
    ├── database.py             # SQLAlchemy engine, session, init_db()
    ├── routers/
    │   ├── chat.py             # Chat endpoints + SSE streaming
    │   ├── analyze.py          # Image upload + vision model
    │   └── advisory.py        # Crop/topic advisory
    ├── services/
    │   ├── ollama_service.py   # Async Ollama HTTP client
    │   ├── prompt_builder.py   # All LLM prompt templates
    │   └── image_service.py    # Pillow preprocessing + base64
    ├── models/
    │   └── database.py         # SQLAlchemy ORM models
    ├── schemas/
    │   └── schemas.py          # Pydantic request/response schemas
    └── requirements.txt
```

---

## Future Improvements

- [ ] **More crops** — Onion, garlic, brinjal, chilli, mustard
- [ ] **Multilingual expansion** — Hindi, Tamil support
- [ ] **Weather integration** — Fetch local weather and factor into irrigation advice
- [ ] **Offline PWA** — Service worker for full offline mobile use
- [ ] **User accounts** — Multi-user support with persistent profiles
- [ ] **Model fine-tuning** — Fine-tune on Bangladesh/India-specific agricultural datasets
- [ ] **Voice input** — Whisper-based speech-to-text for farmers who prefer speaking
- [ ] **Export reports** — Download disease analysis as PDF

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

> Built as a Final Year Project. For serious crop disease outbreaks, always consult your local agricultural extension officer.
