
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from core.llm_utils import get_llm

def split_transcript(transcript : str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )
    return splitter.split_text(transcript)

def summarize(transcript : str) -> str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that summarizes chunks of transcript. Your summary should be concise and in bullet points."),
        ("user", "Summarize this chunk: {chunk}")
    ])
    map_chain = map_prompt | llm | StrOutputParser()
    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"chunk" : chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)
    combined_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that combines and structures summaries from chunks of transcript. Combine them into a single coherent, professional summary in bullet points."),
        ("user", "Combine these chunk summaries: {combined}")
    ])
    combined_chain = combined_prompt | llm | StrOutputParser()
    return combined_chain.invoke({"combined" : combined})

def generate_title(transcript : str) -> str:
    llm = get_llm()
    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text":x}) |
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that generates a concise title for a transcript (max 8 words). Only return the title, nothing else."),
            ("user", "Generate a title for this transcript: {text}")
        ])
        | llm | StrOutputParser()
    )
    return title_chain.invoke(transcript[:2000])