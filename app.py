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
from audio_recorder_streamlit import audio_recorder

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
# Custom CSS — Premium Dark Glassmorphism Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Reset & Globals ── */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    .stApp {
        background: linear-gradient(160deg, #0a0e1a 0%, #111827 40%, #0f172a 100%);
        color: #e2e8f0;
    }
    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }

    /* ── Animated background blobs ── */
    .stApp::before {
        content: '';
        position: fixed;
        top: -200px;
        left: -100px;
        width: 600px;
        height: 600px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        border-radius: 50%;
        filter: blur(80px);
        z-index: 0;
        animation: float1 20s ease-in-out infinite;
    }
    .stApp::after {
        content: '';
        position: fixed;
        bottom: -200px;
        right: -100px;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(168, 85, 247, 0.12) 0%, transparent 70%);
        border-radius: 50%;
        filter: blur(80px);
        z-index: 0;
        animation: float2 25s ease-in-out infinite;
    }
    @keyframes float1 {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(60px, 40px); }
    }
    @keyframes float2 {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(-40px, -60px); }
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: rgba(15, 23, 42, 0.5);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 3rem 2.5rem;
        border-radius: 24px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        text-align: center;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #818cf8, #c084fc, transparent);
    }
    .hero-banner h1 {
        background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin: 0 0 0.5rem 0;
        letter-spacing: -1.5px;
        position: relative;
        z-index: 1;
    }
    .hero-banner .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0;
        position: relative;
        z-index: 1;
    }
    .hero-banner .tech-badges {
        margin-top: 1.2rem;
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }
    .tech-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.04);
        color: #94a3b8;
        transition: all 0.2s ease;
    }
    .tech-badge:hover {
        border-color: rgba(129, 140, 248, 0.3);
        color: #c7d2fe;
        background: rgba(129, 140, 248, 0.08);
    }

    /* ── Input Card ── */
    .input-card {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .input-card h3 {
        color: #e2e8f0;
        font-weight: 700;
        font-size: 1.2rem;
        margin: 0 0 0.3rem 0;
    }
    .input-card .input-desc {
        color: #64748b;
        font-size: 0.85rem;
        margin: 0 0 1.2rem 0;
    }

    /* ── Result Cards ── */
    .result-card {
        background: rgba(15, 23, 42, 0.45);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(129, 140, 248, 0.3), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .result-card:hover {
        border-color: rgba(129, 140, 248, 0.15);
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }
    .result-card:hover::before { opacity: 1; }
    .result-card h3 {
        color: #f1f5f9;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .result-card .content {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.85;
    }

    /* ── Metric Cards (for empty state) ── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    }
    .metric-card {
        flex: 1;
        min-width: 140px;
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(129, 140, 248, 0.2);
        transform: translateY(-2px);
    }
    .metric-card .icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* ── Analysis Complete Badge ── */
    .analysis-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.25);
        color: #34d399;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .meeting-title {
        background: linear-gradient(135deg, #e2e8f0, #f8fafc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.8px;
        margin: 0.8rem 0 0 0;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #64748b;
        background: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #cbd5e1; }
    .stTabs [aria-selected="true"] {
        background: rgba(99, 102, 241, 0.15) !important;
        color: #a5b4fc !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        padding: 0.85rem 1.5rem !important;
        font-size: 1.05rem !important;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
        letter-spacing: 0.3px;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.4);
    }
    .stDownloadButton > button {
        background: rgba(15, 23, 42, 0.6) !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.2rem !important;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stDownloadButton > button:hover {
        background: rgba(99, 102, 241, 0.12) !important;
        border-color: rgba(99, 102, 241, 0.3) !important;
    }

    /* ── Inputs ── */
    .stTextInput input {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #f1f5f9 !important;
        border-radius: 12px !important;
        padding: 0.85rem 1rem !important;
        font-size: 0.95rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        transition: all 0.2s ease;
    }
    .stTextInput input::placeholder {
        color: #475569 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stTextInput input:focus {
        border-color: rgba(129, 140, 248, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.1) !important;
    }
    .stSelectbox > div > div {
        background-color: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #f1f5f9 !important;
        border-radius: 12px !important;
    }
    .stRadio label { color: #cbd5e1 !important; font-weight: 500 !important; }
    .stRadio [data-baseweb="radio"] > div:first-child {
        background-color: rgba(99, 102, 241, 0.2) !important;
    }

    /* ── File Uploader ── */
    .stFileUploader > div {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 2px dashed rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: all 0.2s ease;
    }
    .stFileUploader > div:hover {
        border-color: rgba(129, 140, 248, 0.3) !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(10, 14, 26, 0.95);
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    .sidebar-brand {
        text-align: center;
        padding: 2rem 1rem 1.5rem 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.04);
        margin-bottom: 1.5rem;
    }
    .sidebar-brand h2 {
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
    }
    .sidebar-brand p {
        color: #475569 !important;
        font-size: 0.7rem;
        font-weight: 700;
        margin: 6px 0 0 0;
        text-transform: uppercase;
        letter-spacing: 3px;
    }
    .sidebar-info {
        background: rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.6;
    }
    .sidebar-info strong { color: #94a3b8; }
    .sidebar-footer {
        margin-top: 2rem;
        padding-bottom: 1rem;
        text-align: center;
        color: #334155;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    /* ── Chat ── */
    .stChatMessage {
        background: rgba(15, 23, 42, 0.3) !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
        border-radius: 16px !important;
    }
    .stChatInput > div {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
    }

    /* ── Expander ── */
    .stExpander {
        background: rgba(15, 23, 42, 0.3) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 14px !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa) !important;
        border-radius: 10px;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.04) !important;
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
    <div class="sidebar-brand">
        <h2>🧠 MeetMind</h2>
        <p>AI Meeting Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-info">
        <strong>How it works:</strong><br>
        1. Select an input method on the left<br>
        2. Choose transcription language<br>
        3. Click <strong>Analyze Meeting</strong><br>
        4. Get summaries, action items & more!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-info">
        <strong>Powered by:</strong><br>
        🎤 Whisper · Sarvam AI<br>
        🤖 Gemini · Groq · Mistral<br>
        🔗 LangChain · ChromaDB
    </div>
    """, unsafe_allow_html=True)

    # Download report button (only shown when results exist)
    if st.session_state.result:
        st.markdown("---")
        st.markdown("#### 📄 Export Report")
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
            label="📥 Download Full Report",
            data=report,
            file_name="meetmind_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown("""
    <div class="sidebar-footer">
        Built with ❤️ using Multi-Modal AI
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Hero Banner
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>🧠 MeetMind</h1>
    <p class="subtitle">Transform any meeting recording into actionable insights</p>
    <div class="tech-badges">
        <span class="tech-badge">Whisper</span>
        <span class="tech-badge">Sarvam AI</span>
        <span class="tech-badge">Gemini</span>
        <span class="tech-badge">Groq</span>
        <span class="tech-badge">Mistral</span>
        <span class="tech-badge">LangChain</span>
        <span class="tech-badge">ChromaDB</span>
    </div>
</div>
""", unsafe_allow_html=True)



# ──────────────────────────────────────────────
# Main Dashboard Layout
# ──────────────────────────────────────────────
st.markdown("---")
col_input, col_dashboard = st.columns([1, 2], gap="large")

with col_input:
    st.markdown("""
    <div class="input-card">
        <h3>📥 Analyze a New Meeting</h3>
        <p class="input-desc">Provide your meeting audio below to generate insights.</p>
    </div>
    """, unsafe_allow_html=True)

    input_method = st.radio(
        "Choose input method",
        ["🔗 YouTube", "📁 Upload", "🎙️ Record"],
        horizontal=True,
        label_visibility="collapsed",
    )

    source_url = None
    uploaded_file = None
    recorded_audio = None

    if "YouTube" in input_method:
        source_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/...",
            label_visibility="collapsed"
        )
    elif "Upload" in input_method:
        uploaded_file = st.file_uploader(
            "Upload an audio/video file",
            type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"],
            label_visibility="collapsed"
        )
    else:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(15, 23, 42, 0.4); border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 1rem;">
            <p style="color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem;">Click microphone to record your meeting</p>
        """, unsafe_allow_html=True)
        
        recorded_audio = audio_recorder(text="", icon_size="2x")
        
        if recorded_audio:
            st.success("✅ Audio recorded successfully!")
            st.audio(recorded_audio, format="audio/wav")
            
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    
    language = st.selectbox(
        "🌐 Transcription Language",
        ["English", "Hinglish (Hindi → English)"]
    )
    lang_key = "hinglish" if "Hinglish" in language else "english"

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🚀 Analyze Meeting", type="primary", use_container_width=True)

# ──────────────────────────────────────────────
# Pipeline Execution & Results
# ──────────────────────────────────────────────
with col_dashboard:
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
        if "YouTube" in input_method and not source_url:
            st.error("⚠️ Please enter a YouTube URL.")
            st.stop()
        if "Upload" in input_method and not uploaded_file:
            st.error("⚠️ Please upload an audio file.")
            st.stop()
        if "Record" in input_method and not recorded_audio:
            st.error("⚠️ Please record some audio first.")
            st.stop()

        # Reset previous results and chat
        st.session_state.result = None
        st.session_state.chat_history = []

        # Determine source path
        if "YouTube" in input_method:
            source = source_url
        elif "Upload" in input_method:
            suffix = os.path.splitext(uploaded_file.name)[1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="downloads")
            tmp.write(uploaded_file.read())
            tmp.close()
            source = tmp.name
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="downloads")
            tmp.write(recorded_audio)
            tmp.close()
            source = tmp.name

        # Run the pipeline with progress
        progress_bar = st.progress(0, text="Starting pipeline...")
        status_area = st.empty()

        chunk_size = 29 if lang_key == "hinglish" else 600
        result = {}
        total = len(STEPS)

        try:
            progress_bar.progress(1 / total, text=f"Step 1/{total}: {STEPS[0]}")
            with st.spinner(f"⏳ {STEPS[0]}..."):
                chunks = process_input(source, chunk_length_seconds=chunk_size)

            progress_bar.progress(2 / total, text=f"Step 2/{total}: {STEPS[1]}")
            with st.spinner(f"⏳ {STEPS[1]}..."):
                transcript = transcribe_all(chunks, lang_key)

            progress_bar.progress(3 / total, text=f"Step 3/{total}: {STEPS[2]}")
            with st.spinner(f"⏳ {STEPS[2]}..."):
                title = generate_title(transcript)

            progress_bar.progress(4 / total, text=f"Step 4/{total}: {STEPS[3]}")
            with st.spinner(f"⏳ {STEPS[3]}..."):
                summary = summarize(transcript)

            progress_bar.progress(5 / total, text=f"Step 5/{total}: {STEPS[4]}")
            with st.spinner(f"⏳ {STEPS[4]}..."):
                action_items = extract_action_items(transcript)

            progress_bar.progress(6 / total, text=f"Step 6/{total}: {STEPS[5]}")
            with st.spinner(f"⏳ {STEPS[5]}..."):
                key_decisions = extract_key_decisions(transcript)

            progress_bar.progress(7 / total, text=f"Step 7/{total}: {STEPS[6]}")
            with st.spinner(f"⏳ {STEPS[6]}..."):
                questions = extract_questions(transcript)

            progress_bar.progress(8 / total, text=f"Step 8/{total}: {STEPS[7]}")
            with st.spinner(f"⏳ {STEPS[7]}..."):
                rag_chain = build_rag_chain(transcript)

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

    if st.session_state.result:
        r = st.session_state.result

        # Title area
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="analysis-badge">✓ Analysis Complete</span>
            <h2 class="meeting-title">{r['title']}</h2>
        </div>
        """, unsafe_allow_html=True)

        # Tabs
        tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs([
            "📝 Summary",
            "✅ Actions",
            "🎯 Decisions",
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
                <div class="content" style="color: #64748b; font-size: 0.85rem;">
                    Ask any question about the meeting — powered by RAG retrieval.
                </div>
            </div>
            """, unsafe_allow_html=True)

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask something about the meeting..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        rag_chain = r["rag_chain"]
                        response = rag_chain.invoke(prompt)
                    st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    else:
        # Empty state
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem;">
            <div style="
                width: 70px; height: 70px; margin: 0 auto 1.5rem auto;
                background: rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(99, 102, 241, 0.15);
                border-radius: 18px;
                display: flex; align-items: center; justify-content: center;
                font-size: 2rem;
            ">🎙️</div>
            <h3 style="color: #e2e8f0; font-weight: 700; font-size: 1.3rem; margin-bottom: 0.5rem;">Ready to Analyze</h3>
            <p style="color: #64748b; font-size: 0.9rem; margin: 0 auto 2rem auto; line-height: 1.6;">
                Select an input source on the left to generate insights.
            </p>
            <div class="metric-row" style="justify-content: center;">
                <div class="metric-card">
                    <div class="icon">📝</div>
                    <div class="label">Summary</div>
                </div>
                <div class="metric-card">
                    <div class="icon">✅</div>
                    <div class="label">Actions</div>
                </div>
                <div class="metric-card">
                    <div class="icon">🎯</div>
                    <div class="label">Decisions</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
