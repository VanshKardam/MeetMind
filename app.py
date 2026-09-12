from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import tempfile
import os
import time

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="MeetMind — AI Meeting Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Gradient hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 30%, #4338ca 60%, #6366f1 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(139,92,246,0.3) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }
    .hero-banner p {
        color: #c7d2fe;
        font-size: 1rem;
        margin: 0;
        position: relative;
        z-index: 1;
        font-weight: 400;
    }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0d2e 0%, #1a1744 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e7ff !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #a5b4fc !important;
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Result cards ── */
    .result-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .result-card:hover {
        box-shadow: 0 4px 16px rgba(99,102,241,0.12);
        transform: translateY(-1px);
    }
    .result-card h3 {
        color: #312e81;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .result-card .content {
        color: #374151;
        font-size: 0.92rem;
        line-height: 1.75;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: #4338ca !important;
        color: #ffffff !important;
        border-radius: 8px;
    }

    /* ── Status badges ── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .status-processing {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }
    .status-complete {
        background: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }

    /* ── Chat messages ── */
    .stChatMessage {
        border-radius: 12px !important;
    }

    /* ── Sidebar logo area ── */
    .sidebar-logo {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid rgba(99,102,241,0.2);
        margin-bottom: 1.5rem;
    }
    .sidebar-logo h2 {
        color: #a5b4fc !important;
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .sidebar-logo p {
        color: #6366f1 !important;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 4px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* ── Progress step ── */
    .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 6px 0;
        font-size: 0.85rem;
    }
    .step-done { color: #10b981 !important; }
    .step-active { color: #f59e0b !important; font-weight: 600; }
    .step-pending { color: #6b7280 !important; }

    /* ── Download button ── */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4338ca, #6366f1) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 600 !important;
        width: 100%;
        transition: opacity 0.2s ease;
    }
    .stDownloadButton > button:hover {
        opacity: 0.9;
    }

    /* ── Primary action button ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4338ca, #6366f1) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.7rem 1.5rem !important;
        font-size: 1rem !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(99,102,241,0.4);
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>🧠 MeetMind</h2>
        <p>AI Meeting Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 📥 Input Source")
    input_method = st.radio(
        "Choose input method",
        ["YouTube URL", "Upload Audio File"],
        horizontal=True,
        label_visibility="collapsed",
    )

    source_url = None
    uploaded_file = None

    if input_method == "YouTube URL":
        source_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=...",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload an audio/video file",
            type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"],
        )

    st.markdown("---")
    st.markdown("#### 🌐 Language")
    language = st.selectbox(
        "Select language",
        ["English", "Hinglish (Hindi → English)"],
        label_visibility="collapsed",
    )
    lang_key = "hinglish" if "Hinglish" in language else "english"

    st.markdown("---")

    analyze_clicked = st.button("🚀 Analyze Meeting", type="primary", use_container_width=True)

    # Download report button (only shown when results exist)
    if st.session_state.result:
        st.markdown("---")
        r = st.session_state.result
        report = (
            f"MEETMIND — MEETING REPORT\n"
            f"{'='*50}\n\n"
            f"TITLE: {r['title']}\n\n"
            f"{'='*50}\n"
            f"SUMMARY\n{'='*50}\n{r['summary']}\n\n"
            f"{'='*50}\n"
            f"ACTION ITEMS\n{'='*50}\n{r['action_items']}\n\n"
            f"{'='*50}\n"
            f"KEY DECISIONS\n{'='*50}\n{r['key_decisions']}\n\n"
            f"{'='*50}\n"
            f"QUESTIONS RAISED\n{'='*50}\n{r['questions']}\n\n"
            f"{'='*50}\n"
            f"FULL TRANSCRIPT\n{'='*50}\n{r['transcript']}\n"
        )
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name="meetmind_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ──────────────────────────────────────────────
# Hero Banner
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🧠 MeetMind</h1>
    <p>Transform any meeting recording into actionable insights — powered by Whisper, Sarvam AI, Gemini, Groq, Mistral &amp; LangChain</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Pipeline Execution
# ──────────────────────────────────────────────
STEPS = [
    "Processing audio input",
    "Transcribing audio",
    "Generating title",
    "Summarizing transcript",
    "Extracting action items",
    "Extracting key decisions",
    "Extracting questions",
    "Building RAG knowledge base",
]

if analyze_clicked:
    # Validate input
    if input_method == "YouTube URL" and not source_url:
        st.error("Please enter a YouTube URL.")
        st.stop()
    if input_method == "Upload Audio File" and not uploaded_file:
        st.error("Please upload an audio file.")
        st.stop()

    # Reset previous results and chat
    st.session_state.result = None
    st.session_state.chat_history = []

    # Determine source path
    if input_method == "YouTube URL":
        source = source_url
    else:
        # Save uploaded file to a temp location
        suffix = os.path.splitext(uploaded_file.name)[1]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="downloads")
        tmp.write(uploaded_file.read())
        tmp.close()
        source = tmp.name

    # Run the pipeline with progress
    progress_bar = st.progress(0, text="Starting pipeline...")
    status_area = st.empty()

    chunk_size = 29 if lang_key == "hinglish" else 600
    result = {}
    total = len(STEPS)

    try:
        # Step 1 — Process audio
        progress_bar.progress(1 / total, text=f"Step 1/{total}: {STEPS[0]}")
        with st.spinner(f"⏳ {STEPS[0]}..."):
            chunks = process_input(source, chunk_length_seconds=chunk_size)

        # Step 2 — Transcribe
        progress_bar.progress(2 / total, text=f"Step 2/{total}: {STEPS[1]}")
        with st.spinner(f"⏳ {STEPS[1]}..."):
            transcript = transcribe_all(chunks, lang_key)

        # Step 3 — Title
        progress_bar.progress(3 / total, text=f"Step 3/{total}: {STEPS[2]}")
        with st.spinner(f"⏳ {STEPS[2]}..."):
            title = generate_title(transcript)

        # Step 4 — Summary
        progress_bar.progress(4 / total, text=f"Step 4/{total}: {STEPS[3]}")
        with st.spinner(f"⏳ {STEPS[3]}..."):
            summary = summarize(transcript)

        # Step 5 — Action Items
        progress_bar.progress(5 / total, text=f"Step 5/{total}: {STEPS[4]}")
        with st.spinner(f"⏳ {STEPS[4]}..."):
            action_items = extract_action_items(transcript)

        # Step 6 — Key Decisions
        progress_bar.progress(6 / total, text=f"Step 6/{total}: {STEPS[5]}")
        with st.spinner(f"⏳ {STEPS[5]}..."):
            key_decisions = extract_key_decisions(transcript)

        # Step 7 — Questions
        progress_bar.progress(7 / total, text=f"Step 7/{total}: {STEPS[6]}")
        with st.spinner(f"⏳ {STEPS[6]}..."):
            questions = extract_questions(transcript)

        # Step 8 — RAG
        progress_bar.progress(8 / total, text=f"Step 8/{total}: {STEPS[7]}")
        with st.spinner(f"⏳ {STEPS[7]}..."):
            rag_chain = build_rag_chain(transcript)

        # Done!
        progress_bar.progress(1.0, text="✅ Analysis complete!")
        time.sleep(0.5)
        progress_bar.empty()

        st.session_state.result = {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": key_decisions,
            "questions": questions,
            "rag_chain": rag_chain,
        }
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        st.error(f"❌ Pipeline failed at current step: {e}")
        st.stop()

# ──────────────────────────────────────────────
# Results Dashboard
# ──────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <span style="
            display: inline-block;
            background: linear-gradient(135deg, #4338ca, #6366f1);
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 30px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 1px;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        ">Meeting Analysis Complete</span>
        <h2 style="
            color: #1e1b4b;
            font-size: 1.8rem;
            font-weight: 800;
            margin: 0.5rem 0 0 0;
            letter-spacing: -0.5px;
        ">{r['title']}</h2>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs([
        "📝 Summary",
        "✅ Action Items",
        "🎯 Key Decisions",
        "❓ Questions",
        "📄 Transcript",
        "💬 Chat",
    ])

    with tab_summary:
        st.markdown(f"""
        <div class="result-card">
            <h3>📝 Meeting Summary</h3>
            <div class="content">{r['summary'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

    with tab_actions:
        st.markdown(f"""
        <div class="result-card">
            <h3>✅ Action Items</h3>
            <div class="content">{r['action_items'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

    with tab_decisions:
        st.markdown(f"""
        <div class="result-card">
            <h3>🎯 Key Decisions</h3>
            <div class="content">{r['key_decisions'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

    with tab_questions:
        st.markdown(f"""
        <div class="result-card">
            <h3>❓ Questions Raised</h3>
            <div class="content">{r['questions'].replace(chr(10), '<br>')}</div>
        </div>
        """, unsafe_allow_html=True)

    with tab_transcript:
        st.markdown("""
        <div class="result-card">
            <h3>📄 Full Transcript</h3>
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Click to expand the full transcript", expanded=False):
            st.text(r["transcript"])

    with tab_chat:
        st.markdown("""
        <div class="result-card">
            <h3>💬 Chat with Your Meeting</h3>
            <div class="content" style="color: #6b7280; font-size: 0.85rem;">
                Ask any question about the meeting — powered by RAG retrieval over your transcript.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask something about the meeting..."):
            # Show user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get RAG response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    rag_chain = r["rag_chain"]
                    response = rag_chain.invoke(prompt)
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

else:
    # Empty state — no results yet
    st.markdown("""
    <div style="
        text-align: center;
        padding: 4rem 2rem;
        color: #94a3b8;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🎙️</div>
        <h3 style="color: #64748b; font-weight: 600; margin-bottom: 0.5rem;">No meeting analyzed yet</h3>
        <p style="font-size: 0.95rem; max-width: 400px; margin: 0 auto;">
            Paste a YouTube URL or upload an audio file in the sidebar, then click
            <strong>"🚀 Analyze Meeting"</strong> to get started.
        </p>
    </div>
    """, unsafe_allow_html=True)
