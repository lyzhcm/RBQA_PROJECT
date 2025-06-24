import streamlit as st
from Files_Operator import parse_file, generate_file_id, load_embedding_model
from Database_Operator import (
    delete_file,
    restore_file,
    toggle_file_tag,
    semantic_analysis,
    update_vector_store
)
from AI_Respond import ask_ai
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
import shutil



def init_session():
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = []
    if "deleted_files" not in st.session_state:
        st.session_state.deleted_files = []
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = None
    if "knowledge_vectors" not in st.session_state:
        st.session_state.knowledge_vectors = []
    if "vector_db" not in st.session_state:
        st.session_state.embedding_model_langchain = HuggingFaceEmbeddings(
            model_name="GanymedeNil/text2vec-large-chinese",
            model_kwargs={'device': 'cpu'}
        )
        st.session_state.vector_db = Chroma(
            embedding_function=st.session_state.embedding_model_langchain,
            persist_directory="./chroma_db"
        )
    if "text_splitter" not in st.session_state:
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )


def knowledge_base_section():
    from UI import knowledge_base_section as kb_section
    kb_section()


def qa_interface():
    from UI import qa_interface as qa_func
    qa_func()

def is_model_downloaded(model_name):
    # SentenceTransformer 默认缓存路径
    cache_dir = os.path.expanduser("~/.cache/torch/sentence_transformers")
    model_dir = os.path.join(cache_dir, model_name)
    return os.path.exists(model_dir) and os.path.isdir(model_dir)

def main():
    st.set_page_config(
        page_title="智能文献问答系统",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    # 只依赖本地模型检测，不依赖 session_state
    if "embedding_model" not in st.session_state or st.session_state.embedding_model is None:
        if is_model_downloaded(model_name):
            model = SentenceTransformer(model_name)
            st.session_state.embedding_model = model
        else:
            with st.spinner("正在检测是否已预下载语义模型（约1.2GB，如首次运行需要时间）..."):
                try:
                    model = SentenceTransformer(model_name)
                    st.session_state.embedding_model = model
                except Exception as e:
                    st.error(f"模型下载失败: {str(e)}")
                    return

    init_session()
    st.title("📚 智能文献问答系统")
    st.caption("知识库构建、管理及智能问答平台 | 支持文档处理与语义分析")

    with st.sidebar:
        st.header("🔍 导航菜单")
        page = st.radio("选择功能", ["知识库管理", "智能问答"], horizontal=True)
        st.divider()
        st.subheader("📊 系统概览")
        col1, col2, col3 = st.columns(3)
        col1.metric("知识文档", len(st.session_state.uploaded_files))
        col2.metric("知识片段", len(st.session_state.knowledge_base))
        col3.metric("回收站", len(st.session_state.deleted_files))
        st.divider()
        st.info("""
        **系统功能：**
        1. 知识库构建与管理
        2. 文档解析与处理
        3. 语义分析与理解
        4. 智能问答系统
        5. 文件标记与回收
        """)
        if st.button("清空所有数据", use_container_width=True, type="secondary"):
            # 清空会话状态
            st.session_state.uploaded_files = []
            st.session_state.knowledge_base = []
            st.session_state.conversation = []
            st.session_state.knowledge_vectors = []
            
            # 清空 ChromaDB 数据
            st.session_state.vector_db.delete_collection() 
            st.session_state.vector_db = Chroma(
                embedding_function=st.session_state.embedding_model_langchain,
                persist_directory="./chroma_db"
            )
            
            st.rerun()

    if page == "知识库管理":
        knowledge_base_section()
    else:
        qa_interface()

    with st.expander("🛠️ 调试信息", expanded=False):
        st.json({
            "文件上传数": len(st.session_state.uploaded_files),
            "知识片段数": len(st.session_state.knowledge_base),
            "向量存储数": len(st.session_state.knowledge_vectors) if hasattr(st.session_state.knowledge_vectors, '__len__') else 0,
            "删除文件数": len(st.session_state.deleted_files),
            "对话轮次": len(st.session_state.conversation) // 2
        })


if __name__ == "__main__":
    main()