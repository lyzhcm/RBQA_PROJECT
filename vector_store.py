import streamlit as st
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL_LANGCHAIN, CHROMA_DB_PATH
from typing import List, Dict, Optional

@st.cache_resource
def get_embedding_function():
    """获取LangChain的嵌入函数"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_LANGCHAIN,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )

@st.cache_resource
def get_vector_db():
    """初始化并返回ChromaDB实例"""
    embedding_function = get_embedding_function()
    return Chroma(
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_PATH
    )

def add_texts_to_db(texts: List[str], metadatas: List[Dict]):
    """添加文本和元数据到向量数据库"""
    assert len(texts) == len(metadatas), f"texts({len(texts)})和metadatas({len(metadatas)})长度不一致"
    db = get_vector_db()
    db.add_texts(texts=texts, metadatas=metadatas)
    
    # 更新会话状态（如果需要）
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = db

def search_db(query: str, k: int = 3) -> List:
    """在向量数据库中执行相似性搜索"""
    db = get_vector_db()
    return db.similarity_search(query, k=k)

def delete_from_db_by_source_id(source_id: str):
    """根据source_id元数据删除向量"""
    db = get_vector_db()
    db.delete(where={"source_id": source_id})

def clear_db():
    """清空向量数据库集合"""
    db = get_vector_db()
    db.delete_collection()
    
    # 重置会话状态
    if "vector_db" in st.session_state:
        del st.session_state.vector_db

def load_existing_documents() -> Optional[List[Dict]]:
    """加载向量数据库中现有的文档"""
    try:
        db = get_vector_db()
        collection = db._collection
        if collection is not None:
            return collection.get(include=["metadatas", "documents"])
        return None
    except Exception as e:
        st.error(f"加载现有文档失败: {str(e)}")
        return None