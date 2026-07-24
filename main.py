from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

def run_pipeline(source : str, language : str = "english") -> dict:
    print("\nSTEP 1: Processing input")
    chunk_size = 29 if language.lower() == "hinglish" else 600
    chunks = process_input(source, chunk_length_seconds=chunk_size)

    print("\nSTEP 2: Transcribing audio")
    transcript = transcribe_all(chunks, language)

    print("\nSTEP 3: Generating title")
    title = generate_title(transcript)

    print("\nSTEP 4: Summarizing transcript")
    summary = summarize(transcript)

    print("\nSTEP 5: Extracting action items")
    action_items = extract_action_items(transcript)

    print("\nSTEP 6: Extracting key decisions")
    key_decisions = extract_key_decisions(transcript)

    print("\nSTEP 7: Extracting questions")
    questions = extract_questions(transcript)

    print("\nSTEP 8: Building RAG chain")
    rag_chain = build_rag_chain(transcript)

    return {
        "title" : title,
        "transcript" : transcript,
        "summary" : summary,
        "action_items" : action_items,
        "key_decisions" : key_decisions,
        "questions" : questions,
        "rag_chain" : rag_chain
    }

if __name__ == "__main__":
    # Phase 1 - CLI entry point
    source = input("Enter Youtube URL or local file path : ").strip()
    if not source:
        print("Error: No source provided")
        exit(1)
    language = input("Enter language (english/hindi) : ").strip() or "english"
    result = run_pipeline(source, language)
    
    print("\n" + "="*60)
    print("Title:", result["title"])
    print("\n" + "="*60)
    print("Summary:", result["summary"])
    print("\n" + "="*60)
    print("Action Items:", result["action_items"])
    print("\n" + "="*60)
    print("Key Decisions:", result["key_decisions"])
    print("\n" + "="*60)
    print("Questions:", result["questions"])
    print("\n" + "="*60)

    # Phase 2 - Chat with your meeting via RAG
    print("\n" + "="*60)
    print("Chat with your meeting via RAG")
    print("="*60)
    print("\n" + "="*60)
    print("Type 'exit' to quit")
    print("="*60)

    rag_chain = result["rag_chain"]
    while True:
        question = input("You : ").strip()
        if question.lower() == "exit":
            print("\nExiting...")
            break
        if not question:
            print("You : (no question)")
            continue
        print(f"You : {question}")
        print("MeetMind: ")
        response = ask_question(rag_chain, question)
        print(response)