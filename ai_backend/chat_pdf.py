import os
from typing_extensions import TypedDict
from langgraph.graph import START, MessagesState, StateGraph, END
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
import os
from ai_backend.config import QDRANT_URL, QDRANT_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
from ai_backend.prompts import query_transformation_prompt, llm_prompts
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

class chat(TypedDict):
    user_query: str 
    transformed_query: str 
    context: str
    answer: str
    token_count: int
    similarity_scores: list

data = []

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

dense_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

qdrant = QdrantVectorStore.from_documents(
    data,
    dense_embeddings,
    sparse_embedding=sparse_embeddings,
    retrieval_mode=RetrievalMode.HYBRID,
    url=QDRANT_URL,
    prefer_grpc=True,
    api_key=QDRANT_API_KEY,
    collection_name="annualreport",
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def query_transformation(state: chat):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                query_transformation_prompt,
            ),
            ("human", "{query}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            "query": state["user_query"],
        }
    )
    
    return {
        "transformed_query": response.content,
    }
        
def retrieve_documents(state: chat):
    results = qdrant.similarity_search_with_score(
        state['transformed_query'], k=2
    )
        
    # Extract content as a single string
    context = ""
    similarity_scores = []
    
    for doc, score in results:
        context += doc.page_content + "\n\n"
        similarity_scores.append(score)
    
    # Remove trailing newlines
    context = context.strip()
    
    return {
        "context": context,
        "similarity_scores": similarity_scores,
    }
    
def generate_answer(state: chat):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                llm_prompts,
            ),
            ("human", "{query}, {context}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            "query": state["user_query"],
            "context": state["context"],
        }
    )
    usage_metadata = response.usage_metadata
    token_count = usage_metadata['total_tokens']
    
    return {
        "answer": response.content,
        "token_count": token_count
    } 
    
chat_builder = StateGraph(chat)

chat_builder.add_node("query_transformation", query_transformation)
chat_builder.add_node("retrieve_documents", retrieve_documents)
chat_builder.add_node("generate_answer", generate_answer)

chat_builder.add_edge(START, "query_transformation")
chat_builder.add_edge("query_transformation", "retrieve_documents")
chat_builder.add_edge("retrieve_documents", "generate_answer")
chat_builder.add_edge("generate_answer", END)

chat_llm_pdf = chat_builder.compile()


"""
response = chat_llm_pdf.invoke(
    chat(
        user_query="Can you tell me about the Alumni homecoming?",
        transformed_query="",
        context="",
        answer=""
    )
)

print(response)
"""