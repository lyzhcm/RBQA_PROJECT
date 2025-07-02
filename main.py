#main
import streamlit as st
import pandas as pd
import os
import time
import signal
import psutil
import webbrowser
from UI import knowledge_base_section, qa_interface
from file_registry import FileRegistry
from session_manager import init_session, clear_session
from config import PERSISTENT_UPLOAD_FOLDER, API_KEY
from vector_store import get_vector_count
from streamlit.runtime.scriptrunner import get_script_run_ctx

def nuclear_exit():
    """原子级清除方案"""
    # 1. 用HTML彻底覆盖页面
    st.markdown("""
    <style>body {margin:0;overflow:hidden;}</style>
    <div id="killswitch" style='
        position:fixed;
        top:0;
        left:0;
        width:100vw;
        height:100vh;
        background:#000;
        color:#f00;
        font-family:Arial;
        z-index:99999;
        display:flex;
        flex-direction:column;
        justify-content:center;
        align-items:center;
    '>
        <h1>系统已终止</h1>
        <p>请关闭页面...</p>
    </div>
    <script>
        // 暴力清除DOM元素
        document.body.innerHTML = '';
        document.body.appendChild(document.getElementById('killswitch'));
        
        // 阻止Streamlit重连
        if (window.streamlit) {
            window.streamlit.closeConnection();
        }
        
        // 最终尝试关闭窗口
        setTimeout(() => {
            window.open('','_self').close();
        }, 500);
    </script>
    """, unsafe_allow_html=True)
    
    # 2. 确保页面更新
    time.sleep(0.5)
    
    # 3. 彻底终止进程（Windows/Mac/Linux通用）
    try:
        ctx = get_script_run_ctx()
        if ctx:
            # 先关闭Streamlit的websocket连接
            if hasattr(ctx, '_ws'):
                ctx._ws.close()
            
            # 杀死相关进程树
            pid = os.getpid()
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
    except Exception:
        os._exit(0)

def main():
    st.set_page_config(
        page_title="智能文献问答系统",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 仅在会话首次加载时运行初始化，并显示加载动画
    if "initialized" not in st.session_state:
        progress_container = st.empty()
        with progress_container.container():
            st.subheader("🚀 系统初始化中...")
            init_progress = st.progress(0, text="准备系统环境...")
            
            # init_session() 执行所有耗时的加载操作
            init_session()
            
            init_progress.progress(50, text="AI模型加载完成...")
            time.sleep(0.5)
            init_progress.progress(100, text="准备就绪！")
            time.sleep(0.5)
        
        progress_container.empty()
        st.session_state.initialized = True
        # st.toast(f"当前 API Key: {API_KEY}") # 移除旧的toast提示

    st.title("📚 智能文献问答系统")
    st.caption("知识库构建、管理及智能问答平台 | 支持文档处理与语义分析")

    with st.sidebar:
        st.header("🔍 导航菜单")
        page = st.radio("选择功能", ["知识库管理", "智能问答"], horizontal=True)
        
        st.divider()
        
        # API密钥设置
        st.subheader("🔑 API密钥设置")
        api_key_input = st.text_input(
            "输入您的API密钥",
            type="password",
            value=st.session_state.get("api_key", ""),
            help="在此输入您的API密钥以使用问答功能。"
        )
        if api_key_input:
            st.session_state.api_key = api_key_input

        st.divider()
        st.subheader("📊 系统概览")
        col1, col2 = st.columns(2)
        col1.metric("知识文档", len(st.session_state.get("uploaded_files", [])))
        col2.metric("回收站", len(st.session_state.get("deleted_files", [])))
        st.divider()
        st.info("""
        **系统功能：**
        1. 知识库构建与管理
        2. 文档解析与处理
        3. 语义分析与理解
        4. 智能问答系统
        5. 文件标记与回收
        """)
        if st.button("安全退出系统"):
            nuclear_exit()
        if st.button("清空所有数据", use_container_width=True, type="secondary"):
            clear_session()
            # 清空后，需要重置初始化标志，以便下次可以重新加载
            if 'initialized' in st.session_state:
                del st.session_state['initialized']
            st.rerun()

    if page == "知识库管理":
        knowledge_base_section()
    else:
        qa_interface()

    with st.expander("🛠️ 调试信息", expanded=False):
        st.json({
            "文件上传数": len(st.session_state.get("uploaded_files", [])),
            "向量存储数": get_vector_count(),
            "删除文件数": len(st.session_state.get("deleted_files", [])),
            "对话轮次": len(st.session_state.get("conversation", [])) if isinstance(st.session_state.get("conversation"), list) else 0
        })


if __name__ == "__main__":
    main()