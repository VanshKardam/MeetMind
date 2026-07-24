from dotenv import load_dotenv
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all


if __name__ == "__main__":
    source = "https://www.youtube.com/watch?v=F2dFAk5fpq0"
    language = "english" # Change to "hinglish" to use Sarvam AI
    
    # Sarvam AI has a max 30s limit, Whisper handles 10m easily
    chunk_size = 29 if language.lower() == "hinglish" else 600
    
    chunks = process_input(source, chunk_length_seconds=chunk_size)
    transcript = transcribe_all(chunks, language=language)
    print(transcript)