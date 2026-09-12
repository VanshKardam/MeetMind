
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.vector_store import build_vector_store, load_vector_store, get_retriever

from core.llm_utils import get_llm

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def build_rag_chain(transcript : str):
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k = 3)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """
        You are an expert meeting analyst assistant that answers questions based on provided context from a meeting transcript.
        
        Rules: 
        1. Answer ONLY based on the context below. Do not use any external knowledge.
        2. If the answer is not in the context, respond with: "Information not available in the meeting transcript."
        3. If the context is empty or irrelevant, say the same.
        4. Keep answers concise but comprehensive based on the context.
        """),
        ("user", """Question: {question}
        Context: {context}"""),
    ])

    # Full LCEL Rag pipeline
    rag_chain = (
        {
            "context" : retriever | RunnableLambda(format_docs),
            "question" : RunnablePassthrough(),
        }
        | prompt | llm | StrOutputParser()
    )

    return rag_chain

def load_rag_chain():
    """
    Load the RAG chain from the vector store.
    """
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=3)
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert meeting analyst assistant that answers questions based on provided context from a meeting transcript.
        
        Rules: 
        1. Answer ONLY based on the context below. Do not use any external knowledge.
        2. If the answer is not in the context, respond with: "Information not available in the meeting transcript."
        3. If the context is empty or irrelevant, say the same.
        4. Keep answers concise but comprehensive based on the context.
        """),
        ("user", """Question: {question}
        Context: {context}"""),
    ])
    
    rag_chain = (
        {
            "context" : retriever | RunnableLambda(format_docs),
            "question" : RunnablePassthrough(),
        }
        | prompt | llm | StrOutputParser()
    )
    return rag_chain

def ask_question(rag_chain, question : str) -> str:
    print("\n" + "="*60)
    print("Question: ", question)
    print("="*60)
    print("Thinking...")
    result = rag_chain.invoke(question) # type: ignore
    print("\nAnswer:", result)
    return result