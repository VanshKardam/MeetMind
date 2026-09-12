from dotenv import load_dotenv
load_dotenv()

from core.llm_utils import get_llm
from core.summarize import summarize
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

def test_pipeline():
    transcript = """
    John: Hi everyone, thanks for joining the meeting. Today we are going to discuss the launch of our new product.
    Sarah: Great. I have prepared the marketing materials. We need to decide on the launch date.
    John: How about next Monday?
    Sarah: Yes, that works. But I need approval for the $500 ad budget.
    John: Approved. Please set up the campaigns by Friday.
    """
    
    print("--- Summary ---")
    print(summarize(transcript))
    
    print("\n--- Action Items ---")
    print(extract_action_items(transcript))
    
    print("\n--- Decisions ---")
    print(extract_key_decisions(transcript))
    
    print("\n--- Questions ---")
    print(extract_questions(transcript))

    print("\n--- RAG Engine ---")
    rag_chain = build_rag_chain(transcript)
    result = ask_question(rag_chain, "What is the launch date?")

if __name__ == "__main__":
    test_pipeline()
