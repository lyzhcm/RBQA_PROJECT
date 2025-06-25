#session_manager
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from vector_store import get_vector_db, clear_db, load_existing_documents
from config import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    EMBEDDING_MODEL_SENTENCE_TRANSFORMER,
    PERSISTENT_UPLOAD_FOLDER
)
from file_parser import parse_file, generate_file_id
from file_registry import FileRegistry
from pathlib import Path
import os

def init_session():
    """初始化所有会话状态变量"""
    # 确保上传目录存在
    os.makedirs(PERSISTENT_UPLOAD_FOLDER, exist_ok=True)
    
    # 基础会话状态
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = []
    if "deleted_files" not in st.session_state:
        st.session_state.deleted_files = []
    
    # 初始化模型
    if "embedding_model" not in st.session_state:
        try:
            st.session_state.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL_SENTENCE_TRANSFORMER
            )
        except Exception as e:
            st.error(f"加载嵌入模型失败: {str(e)}")
            st.session_state.embedding_model = None

    # 初始化文本分割器
    if "text_splitter" not in st.session_state:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
    
    # 初始化向量数据库
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = get_vector_db()
    
    # 从持久化存储加载文件
    # 此逻辑仅在会话中尚未填充文件列表时运行。
    # 这能处理初次启动和会话清除后的重新加载。
    if "uploaded_files" not in st.session_state or not st.session_state.uploaded_files:
        st.session_state.uploaded_files = []
        st.session_state.knowledge_base = []  # 重置以防止重复加载
        registry = FileRegistry.load()
        
        # 加载已注册文件到知识库
        for file_id, file_info in registry.items():
            filepath = Path(file_info["filepath"])
            if filepath.exists():
                # 将文件信息添加到会话状态
                st.session_state.uploaded_files.append({
                    "id": file_id,
                    "name": file_info["filename"],
                    "type": file_info["filename"].split(".")[-1],
                    "local_path": str(filepath),
                    "upload_time": file_info["timestamp"],
                    "tags": ["持久化"]
                })
                
                # 加载文件内容并处理
                with open(filepath, "rb") as f:
                    content = parse_file(f)
                
                if content:
                    # 检查向量数据库中是否已存在该文件的块
                    existing_docs = st.session_state.vector_db.get(where={"source_id": file_id})
                    
                    # 无论是否存在于DB，都加载到内存知识库以供UI显示
                    chunks = st.session_state.text_splitter.split_text(content)
                    for chunk in chunks:
                        st.session_state.knowledge_base.append({
                            "source": file_info["filename"],
                            "source_id": file_id,
                            "content": chunk,
                            "type": file_info["filename"].split(".")[-1]
                        })

                    # 如果DB中不存在，则添加入库
                    if not (existing_docs and existing_docs.get('ids')):
                        metadatas = [{
                            "source": file_info["filename"],
                            "source_id": file_id,
                            "type": file_info["filename"].split(".")[-1],
                            "upload_time": file_info["timestamp"]
                        } for _ in chunks]
                        
                        st.session_state.vector_db.add_texts(
                            texts=chunks,
                            metadatas=metadatas
                        )

def clear_session():
    """清除会话和向量数据库"""
    st.session_state.uploaded_files = []
    st.session_state.knowledge_base = []
    st.session_state.conversation = []
    st.session_state.deleted_files = []
    
    # 清除持久化存储
    registry = FileRegistry.load()
    for file_id, file_info in registry.items():
        try:
            if Path(file_info["filepath"]).exists():
                Path(file_info["filepath"]).unlink()
        except Exception as e:
            st.error(f"删除文件失败: {file_info['filepath']} - {str(e)}")
    
    # 清除注册表和向量数据库
    FileRegistry.save({})
    clear_db()