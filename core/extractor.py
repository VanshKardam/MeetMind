# Actionable items, decision, questions


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


from core.llm_utils import get_llm

def build_chain(system_prompt : str):
    llm = get_llm(temperature=0.1)
    return (RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) | ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ]) | llm | StrOutputParser())

def extract_action_items(transcript: str) -> str:
    system_prompt = """
    You are an expert meeting analyst. From the meeting transcript, your job is to extract all the action items from the transcript For each provide:\n
    1. Task description (who did what or who is supposed to do what)\n
    2. Person responsible\n
    3. Deadline if mentioned, else write "Not mentioned"\n
    Return ONLY the action items, nothing else. If none found say 'No action items found' and end.
    Use bullet points.\n
    """
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)

def extract_key_decisions(transcript: str) -> str:
    system_prompt = """
    You are an expert meeting analyst. From the meeting transcript, your job is to extract all the key decisions made during the meeting. For each provide:\n
    1. The decision made\n
    2. Context or reason for the decision if mentioned\n
    Return ONLY the decisions, nothing else. If none found say 'No key decisions found' and end.
    Use bullet points.\n
    """
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)

def extract_questions(transcript: str) -> str:
    system_prompt = """
    You are an expert meeting analyst. From the meeting transcript, your job is to extract all the important questions raised during the meeting. For each provide:\n
    1. The question asked\n
    2. Who asked it (if identifiable, else write "Unknown")\n
    3. The answer provided (if answered, else write "Unanswered")\n
    Return ONLY the questions, nothing else. If none found say 'No questions found' and end.
    Use bullet points.\n
    """
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)