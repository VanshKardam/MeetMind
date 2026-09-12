# MeetMind: Local RAG Pipeline for Audio Intelligence

![Status: Completed](https://img.shields.io/badge/Status-Completed-success)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

## 📌 Overview

**MeetMind** is a production-ready, privacy-focused productivity tool designed to solve the problem of lost value in professional meetings. While commercial tools exist, they are often expensive and raise data privacy concerns. 

MeetMind solves this by providing a highly capable offline audio transcription and local Retrieval-Augmented Generation (RAG) pipeline. It transcribes video/audio files, extracts smart summaries and action items, and allows users to have a contextual chat with their historical meeting files. 

## ✨ Core Features

*   **Premium Streamlit Dashboard:** A beautiful, dark-themed responsive UI for executing pipelines and interacting with results across a 6-tab dashboard.
*   **Flexible Data Ingestion:** Seamlessly processes YouTube URLs or uploaded local audio/video files (MP4, MP3, WAV).
*   **Dual-Engine Transcription:** 
    *   **English:** Runs OpenAI Whisper locally on the machine to eliminate transcription API costs and protect sensitive data. 
    *   **Hinglish:** Uses Sarvam AI's specialized API with automated 29-second audio chunking for precise regional language transcription.
*   **Automated Intelligence:** Handles smart bulleted summaries, extraction of action items, key decisions, and questions using a robust LLM fallback system (Gemini 1.5 Flash → Groq Llama-3 → Mistral Small) powered by LangChain.
*   **RAG-Powered Meeting Q&A:** Employs a full Retrieval-Augmented Generation workflow allowing users to ask specific semantic questions about what was discussed in the meeting.
*   **Seamless Reporting:** Instant text export capabilities for downloading generated meeting notes and action items.

## 🏗️ Technical Architecture

MeetMind is built with a focus on modularity, leveraging a mix of local edge-computing for heavy workloads and free-tier APIs for generation.

### The Tech Stack

*   **User Interface:** Streamlit (Clean, interactive Python-based web dashboard)
*   **Audio Processing:** `yt-dlp` + `ffmpeg` + `pydub`
*   **Transcription Engine:** Local OpenAI Whisper + Sarvam AI
*   **LLM Generation:** Robust Fallback System using Gemini 1.5 Flash, Groq (Llama-3 70B), and Mistral Small for summarization and precise task extraction.
*   **Embeddings:** Mistral Embed API (`mistral-embed`)
*   **Vector Database:** ChromaDB (Local persistent client)
*   **Orchestration:** LangChain LCEL (LangChain Expression Language for composable pipelines)

## 🚀 Installation & Usage

1. **Clone the repository and enter the directory:**
   ```bash
   git clone https://github.com/your-username/MeetMind.git
   cd MeetMind
   ```

2. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GOOGLE_API_KEY="your_google_api_key_here"
   GROQ_API_KEY="your_groq_api_key_here"
   MISTRAL_API_KEY="your_mistral_api_key_here"
   SARVAM_API_KEY="your_sarvam_api_key_here"
   WHISPER_MODEL="small"
   SARVAM_STT_MODEL="saaras:v2.5"
   ```

3. **Install Dependencies:**
   Ensure you have Python 3.10+ installed. Then install the required packages:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   *(Note: You must have FFmpeg installed on your system for audio processing).*

4. **Run the Application:**
   For Windows users, we have provided a convenient launcher script. Simply run:
   ```bash
   .\run.bat
   ```
   Alternatively, you can manually start the Streamlit server:
   ```bash
   streamlit run app.py
   ```

---
*Developed as a showcase of Multi-Modal AI and Edge-Computing architectures.*