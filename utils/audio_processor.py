from pytubefix import YouTube
from pydub import AudioSegment
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_youtube_audio(url : str) -> str:
    yt = YouTube(url, client='WEB')
    stream = yt.streams.get_audio_only()
    filename = stream.download(output_path=DOWNLOAD_DIR)
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