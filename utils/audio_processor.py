import yt_dlp
from pydub import AudioSegment
import os
import warnings

# Suppress harmless pydub syntax warnings on Python 3.12+
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pydub")

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url : str) -> str:
    """Download audio from YouTube using yt-dlp with nodejs runtime."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        # Use nodejs as JS runtime (deno is not available on Streamlit Cloud)
        'js_runtimes': {'nodejs': {}},
        # Mimic a real browser to avoid 403 on datacenter IPs
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
    return filename

def convert_to_wav(input_path : str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000) #16khz monoaural
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error converting {input_path} to WAV: {e}")
        return None

def chunk_audio(wav_path : str, chunk_length_seconds : int = 600) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_length_seconds * 1000 #milliseconds
    chunks = []
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks

def process_input(source : str, chunk_length_seconds : int = 600) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        downloaded_path = download_youtube_audio(source)
        # Pass the downloaded file through the mono converter
        print("Converting YouTube audio to 16kHz mono...")
        wav_path = convert_to_wav(downloaded_path)
    else:
        print("Detected local file. Converting to 16kHz mono WAV...")
        wav_path = convert_to_wav(source)

    if not wav_path:
        raise ValueError("Failed to process audio input")
    
    print("Audio prepared. Chunking...")
    chunks = chunk_audio(wav_path, chunk_length_seconds)

    print(f"Generated {len(chunks)} audio chunks.") 
    return chunks