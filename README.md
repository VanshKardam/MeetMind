# MeetMind: Local RAG Pipeline for Audio Intelligence

![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-yellow)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Local Compute](https://img.shields.io/badge/Privacy-100%25%20Local-success)

> 🚧 **Note:** This project is currently in active development. The core RAG and Audio processing pipelines are implemented, and UI integration is currently underway.

## 📌 Overview

**MeetMind** is a production-ready, privacy-focused productivity tool designed to solve the problem of lost value in professional meetings. While commercial tools (like Otter.ai) exist, they are often expensive and raise severe data privacy concerns by uploading sensitive corporate audio to third-party servers.

MeetMind solves this by providing a **100% offline audio transcription and local Retrieval-Augmented Generation (RAG) pipeline**. It transcribes video/audio files locally, extracts smart summaries and action items, and allows users to have a contextual chat with their historical meeting files—all without audio ever leaving their local machine.

## ✨ Core Features

*   **Flexible Data Ingestion:** Seamlessly processes YouTube URLs or uploaded local audio/video files (MP4, MP3, WAV).
*   **Privacy-First Transcription:** Runs OpenAI Whisper locally on the machine to eliminate transcription API costs and protect sensitive data. Supports English, Hindi, and Hinglish.
*   **Automated Intelligence:** Handles Hindi-to-English translation, smart bulleted summaries, and automated extraction of action items using Mistral AI.
*   **RAG-Powered Meeting Q&A:** Employs a full Retrieval-Augmented Generation workflow allowing users to ask specific semantic questions about what was discussed across multiple meetings.
*   **Seamless Reporting:** Instant PDF export capabilities for downloading generated meeting notes and action items.

## 🏗️ Technical Architecture

MeetMind is built with a focus on modularity, zero-cost operation, and maximum privacy, leveraging a mix of local edge-computing for heavy workloads and free-tier APIs for generation.

### The Tech Stack

*   **Audio Processing:** `yt-dlp` + `ffmpeg` (Downloading and converting video inputs)
*   **Transcription Engine:** Local OpenAI Whisper (`base`/`small` models for CPU optimization)
*   **LLM Generation:** Mistral AI (Free API Tier) for summarization and precise task extraction.
*   **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local generation for text chunks)
*   **Vector Database:** ChromaDB (Local persistent client for semantic search)
*   **Orchestration:** LangChain LCEL (LangChain Expression Language for composable pipelines)
*   **User Interface:** Streamlit (Clean, interactive Python-based web dashboard)

### Pipeline Flow

1.  **Ingestion:** Audio/Video is downloaded or uploaded.
2.  **Processing:** `ffmpeg` normalizes the audio stream.
3.  **Transcription:** Local Whisper transcribes the audio into text, generating timestamps.
4.  **Vectorization:** LangChain splits the long transcript into overlapping semantic chunks. HuggingFace models embed these chunks into ChromaDB.
5.  **Retrieval & Generation:** User queries the UI. The query is embedded, relevant transcript chunks are retrieved from ChromaDB, and passed to Mistral AI via LCEL to generate a highly contextual answer.

## 🚀 Installation & Usage (Coming Soon)

*Instructions for setting up the virtual environment, installing dependencies (including ffmpeg), and running the Streamlit app will be published upon the v1.0 release.*

---
*Developed as a showcase of Multi-Modal AI and Edge-Computing architectures.*
