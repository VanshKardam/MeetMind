import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_core.runnables import chain

def get_llm(temperature=0.3):
    """
    Returns an LCEL Runnable that implements a fallback mechanism across 
    Gemini, Groq, and Mistral models. It includes a delay to avoid rate limit 
    errors before falling back to the next model.
    """
    models = []
    
    # 1. Gemini (Best Performance, generous free tier)
    if os.getenv("GOOGLE_API_KEY"):
        models.append(ChatGoogleGenerativeAI(
            model="gemini-flash-latest",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature
        ))
    
    # 2. Groq (Very fast, excellent Llama3 performance)
    if os.getenv("GROQ_API_KEY"):
        models.append(ChatGroq(
            model="groq/compound",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature
        ))

    # 3. Mistral (Solid fallback)
    if os.getenv("MISTRAL_API_KEY"):
        models.append(ChatMistralAI(
            model="mistral-small-latest",
            mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=temperature
        ))
        
    if not models:
        raise ValueError("No LLM API keys configured. Please add GOOGLE_API_KEY, GROQ_API_KEY, or MISTRAL_API_KEY to your .env file or Streamlit secrets.")
    
    @chain
    def llm_with_fallback(prompt):
        for i, model in enumerate(models):
            try:
                if i > 0:
                    # Include a 2-second delay before fallback to help mitigate rate limits
                    time.sleep(2)
                return model.invoke(prompt)
            except Exception as e:
                print(f"\n[Warning] Model {model.__class__.__name__} failed: {e}. Falling back to next model...")
                if i == len(models) - 1:
                    # If all models fail, raise the last exception
                    raise e
                    
    return llm_with_fallback
