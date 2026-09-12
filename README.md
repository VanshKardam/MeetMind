# 🧠 MeetMind — AI Meeting Assistant

![Status: Completed](https://img.shields.io/badge/Status-Completed-success)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

> Transform any meeting recording into actionable insights — summaries, action items, key decisions, and an interactive Q&A chatbot, all powered by multi-modal AI.

---

## 📌 Overview

**MeetMind** is a production-ready AI productivity tool that eliminates the pain of lost meeting context. Paste a YouTube URL, upload a recording, or **record live from your browser** — and MeetMind will transcribe, summarize, extract action items, and let you chat with your meeting using RAG.

It is deployed on **Streamlit Community Cloud** and uses a multi-model LLM fallback strategy (Gemini → Groq → Mistral) to maximize uptime on free-tier APIs.

---

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| 🎙️ **Live Audio Recording** | Record meetings directly in the browser using the built-in microphone recorder |
| 🔗 **YouTube Ingestion** | Paste any YouTube URL — audio is downloaded via `yt-dlp` with datacenter-safe settings |
| 📁 **File Upload** | Upload local audio/video files (MP3, WAV, M4A, MP4, WebM, OGG) |
| ✍️ **Dual-Engine Transcription** | English via local **OpenAI Whisper**; Hinglish via **Sarvam AI** API |
| 📝 **Smart Summarization** | Generates concise, bulleted meeting summaries using LangChain LCEL pipelines |
| ✅ **Action Item Extraction** | Automatically identifies tasks, deadlines, and assignees |
| 🎯 **Key Decision Extraction** | Pulls out final decisions and consensus points |
| ❓ **Question Detection** | Flags unresolved questions raised during the meeting |
| 💬 **RAG Chat** | Ask any question about the meeting — answers are grounded in the transcript via ChromaDB retrieval |
| 📥 **Report Export** | One-click download of a full meeting report (TXT) |
| 🎨 **Premium Dark UI** | Glassmorphic dark theme with animated gradients, hover effects, and modern typography |

---

## 🏗️ Technical Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                     │
│   ┌──────────┐   ┌──────────┐   ┌──────────────────────┐    │
│   │ YouTube  │   │  Upload  │   │  🎙️ Record (Browser) │    │
│   └────┬─────┘   └────┬─────┘   └──────────┬───────────┘    │
│        └───────────────┼────────────────────┘                │
│                        ▼                                      │
│              utils/audio_processor.py                         │
│        (yt-dlp download → pydub → 16kHz mono WAV)            │
│                        ▼                                      │
│               core/transcriber.py                             │
│      (Whisper for English │ Sarvam AI for Hinglish)           │
│                        ▼                                      │
│          ┌─────────────┼─────────────┐                        │
│          ▼             ▼             ▼                        │
│   core/summarize  core/extractor  core/rag_engine             │
│   (Summary)       (Actions,       (ChromaDB + Mistral         │
│                    Decisions,      Embeddings → RAG Chat)      │
│                    Questions)                                  │
│          └─────────────┼─────────────┘                        │
│                        ▼                                      │
│            core/llm_utils.py (Fallback LLM Chain)             │
│         Gemini Flash → Groq Compound → Mistral Small          │
└──────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit (Glassmorphic dark theme with custom CSS) |
| **Audio Processing** | `yt-dlp` + `ffmpeg` + `pydub` |
| **In-Browser Recording** | `audio-recorder-streamlit` |
| **Transcription** | OpenAI Whisper (local, CPU) + Sarvam AI (API) |
| **LLM Generation** | Gemini Flash → Groq Compound → Mistral Small (fallback chain) |
| **Embeddings** | Mistral Embed API (`mistral-embed`) |
| **Vector Database** | ChromaDB (local persistent storage) |
| **Orchestration** | LangChain LCEL (composable pipelines) |

---

## 📂 Project Structure

```
MeetMind/
├── app.py                    # Streamlit UI + pipeline orchestration
├── core/
│   ├── llm_utils.py          # Multi-model LLM fallback chain
│   ├── transcriber.py        # Whisper + Sarvam transcription engine
│   ├── summarize.py          # LangChain summarization pipeline
│   ├── extractor.py          # Action items, decisions, questions extraction
│   ├── rag_engine.py         # ChromaDB RAG pipeline for meeting Q&A
│   └── vector_store.py       # ChromaDB vector store management
├── utils/
│   └── audio_processor.py    # yt-dlp download, format conversion, chunking
├── requirements.txt          # Python dependencies
├── packages.txt              # System-level apt packages (for Streamlit Cloud)
├── run.bat                   # Windows launcher script
├── .env                      # API keys (not committed)
└── README.md
```

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.10+
- FFmpeg installed on your system ([download](https://ffmpeg.org/download.html))
- API keys for at least one LLM provider

### 1. Clone the Repository

```bash
git clone https://github.com/VanshKardam/MeetMind.git
cd MeetMind
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY="your_google_api_key"
GROQ_API_KEY="your_groq_api_key"
MISTRAL_API_KEY="your_mistral_api_key"
SARVAM_API_KEY="your_sarvam_api_key"
WHISPER_MODEL="small"
SARVAM_STT_MODEL="saaras:v2.5"
```

> **Note:** You need at least one of `GOOGLE_API_KEY`, `GROQ_API_KEY`, or `MISTRAL_API_KEY`. The app uses a fallback chain — if one model hits rate limits, it automatically switches to the next.

### 5. Run the Application

```bash
# Windows (one-click launcher)
.\run.bat

# Or manually
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deployment (Streamlit Community Cloud)

MeetMind is designed for seamless deployment on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push your code to a GitHub repository.
2. Add your API keys as **Secrets** in the Streamlit Cloud dashboard (Settings → Secrets), using the same format as the `.env` file.
3. The `packages.txt` file automatically installs `ffmpeg`, `nodejs`, and `npm` on the cloud server.
4. YouTube downloads use `yt-dlp` with the `mediaconnect` player client to bypass PoToken/SABR restrictions on datacenter IPs.

---

## 🔄 LLM Fallback Strategy

MeetMind uses a resilient multi-model fallback chain to maximize uptime on free-tier APIs:

```
Gemini Flash (Primary) → Groq Compound (Secondary) → Mistral Small (Tertiary)
```

If the primary model returns a rate limit error (429) or is unavailable, the system automatically retries with the next model after a 2-second cooldown. This ensures the pipeline rarely fails, even under heavy free-tier usage.

---

## 🛠️ Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Whisper runs locally (CPU)** | Zero API cost for English transcription; complete data privacy |
| **Lazy model loading (Singleton)** | Whisper loads once on first chunk and stays in memory — subsequent chunks are instant |
| **yt-dlp over pytubefix** | More robust against YouTube's evolving anti-bot protections (PoToken, SABR) |
| **mediaconnect player client** | Bypasses JavaScript runtime requirements on headless cloud servers |
| **ChromaDB for RAG** | Lightweight, local vector store — no external database needed |
| **LangChain LCEL pipelines** | Composable, readable chain syntax for summarization and extraction |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using Multi-Modal AI · Whisper · Gemini · Groq · Mistral · LangChain · ChromaDB
</p>