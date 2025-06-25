import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_LANGCHAIN, CHROMA_DB_PATH

@st.cache_resource
def get_embedding_function():
    """Get the embedding function for LangChain."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_LANGCHAIN,
        model_kwargs={'device': 'cpu'}
    )

def get_vector_db():
    """Initializes and returns the ChromaDB instance."""
    embedding_function = get_embedding_function()
    return Chroma(
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_PATH
    )

def add_texts_to_db(texts, metadatas):
    """Adds texts and their metadatas to the vector database."""
    if "vector_db" in st.session_state and st.session_state.vector_db:
        st.session_state.vector_db.add_texts(texts=texts, metadatas=metadatas)

def search_db(query, k=3):
    """Performs similarity search in the vector database."""
    if "vector_db" in st.session_state and st.session_state.vector_db:
        return st.session_state.vector_db.similarity_search(query, k=k)
    return []

def delete_from_db_by_source_id(source_id: str):
    """Deletes vectors from the database based on the source_id metadata."""
    if "vector_db" in st.session_state and st.session_state.vector_db:
        st.session_state.vector_db.delete(where={"source_id": source_id})

def clear_db():
    """Deletes the entire collection from the vector database."""
    if "vector_db" in st.session_state and st.session_state.vector_db:
        st.session_state.vector_db.delete_collection()
        st.session_state.vector_db = get_vector_db()
