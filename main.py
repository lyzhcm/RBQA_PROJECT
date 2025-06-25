import streamlit as st
import pandas as pd
import os
import time
from UI import knowledge_base_section, qa_interface
from session_manager import init_session, clear_session

def main():
    st.set_page_config(
        page_title="智能文献问答系统",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Session state is initialized once, including models.
    progress_container = st.empty()

    # 步骤1: 初始化会话状态
    with progress_container.container():
        st.subheader("🚀 系统初始化中...")
        init_progress = st.progress(0, text="准备系统环境")
        init_session()  # 您的初始化函数
        init_progress.progress(30, text="加载AI模型")

    # 步骤2: 渲染主界面
    with progress_container.container():
        init_progress.progress(60, text="构建用户界面")
        st.title("📚 智能文献问答系统")
        st.caption("知识库构建、管理及智能问答平台 | 支持文档处理与语义分析")

    # 步骤3: 完成初始化
    with progress_container.container():
        init_progress.progress(100, text="准备就绪！")
        time.sleep(0.5)  # 让用户看到完成状态
        progress_container.empty()  # 隐藏进度条容器

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
            clear_session()
            st.rerun()

    if page == "知识库管理":
        knowledge_base_section()
    else:
        qa_interface()

    with st.expander("🛠️ 调试信息", expanded=False):
        st.json({
            "文件上传数": len(st.session_state.uploaded_files),
            "知识片段数": len(st.session_state.knowledge_base),
            "向量存储数": st.session_state.vector_db._collection.count() if "vector_db" in st.session_state and st.session_state.vector_db._collection is not None else 0,
            "删除文件数": len(st.session_state.deleted_files),
            "对话轮次": len(st.session_state.conversation) // 2
        })


if __name__ == "__main__":
    main()