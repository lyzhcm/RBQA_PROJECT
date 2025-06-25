import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from vector_db_operator import get_vector_db, clear_db
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_SENTENCE_TRANSFORMER

def init_session():
    """Initializes all session state variables."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = []
    if "deleted_files" not in st.session_state:
        st.session_state.deleted_files = []
    
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = SentenceTransformer(EMBEDDING_MODEL_SENTENCE_TRANSFORMER)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = get_vector_db()

    if "text_splitter" not in st.session_state:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )

def clear_session():
    """Clears all data from the session and the vector database."""
    st.session_state.uploaded_files = []
    st.session_state.knowledge_base = []
    st.session_state.conversation = []
    st.session_state.deleted_files = []
    
    clear_db()
