import streamlit as st
import time
from datetime import datetime
import os
import pandas as pd
import json
from docx import Document
from PyPDF2 import PdfReader
import pptx
import tempfile


# 初始化会话状态
def init_session():
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = []


# 文件解析函数
def parse_file(file):
    content = ""
    file_type = file.name.split(".")[-1].lower()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        if file_type == "txt":
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()

        elif file_type == "pdf":
            reader = PdfReader(tmp_path)
            for page in reader.pages:
                content += page.extract_text() + "\n"

        elif file_type == "docx":
            doc = Document(tmp_path)
            for para in doc.paragraphs:
                content += para.text + "\n"

        elif file_type == "pptx":
            prs = pptx.Presentation(tmp_path)
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        content += shape.text + "\n"

        os.unlink(tmp_path)
        return content

    except Exception as e:
        st.error(f"解析文件时出错: {str(e)}")
        return None


# 知识库管理界面
def knowledge_base_section():
    st.header("📚 知识库构建与管理")

    # 文件上传区域
    uploaded_files = st.file_uploader(
        "上传知识文档 (支持多种格式)",
        type=["txt", "pdf", "docx", "pptx", "md"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # 处理新上传的文件
        for file in uploaded_files:
            if file not in st.session_state.uploaded_files:
                with st.spinner(f"解析文件: {file.name}..."):
                    content = parse_file(file)

                    if content:
                        st.session_state.uploaded_files.append({
                            "name": file.name,
                            "type": file.type,
                            "content": content,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                        # 分割内容为知识片段
                        chunks = split_into_chunks(content)
                        for i, chunk in enumerate(chunks):
                            st.session_state.knowledge_base.append({
                                "source": file.name,
                                "content": chunk,
                                "type": file.type.split("/")[-1]
                            })

        # 显示上传文件列表
        st.subheader("已上传文档")
        files_df = pd.DataFrame([
            {
                "文件名": f["name"],
                "类型": f["type"],
                "上传时间": f["upload_time"]
            }
            for f in st.session_state.uploaded_files
        ])
        st.dataframe(files_df)

        # 查看知识库内容
        with st.expander("查看知识库内容"):
            for item in st.session_state.knowledge_base[:5]:  # 显示前5个
                st.caption(f"来源: {item['source']} | 类型: {item['type']}")
                st.text(item["content"][:150] + "..." if len(item["content"]) > 150 else item["content"])
                st.divider()


# 将文档内容分割为适合处理的片段
def split_into_chunks(text, chunk_size=200):
    chunks = []
    words = text.split()

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


# 问答界面
def qa_interface():
    st.header("💬 智能问答系统")

    # 显示历史对话
    if st.session_state.conversation:
        st.subheader("对话历史")
        for i, message in enumerate(st.session_state.conversation):
            role = "🕵️‍♂️ 用户" if i % 2 == 0 else "🤖 系统"
            with st.chat_message(name=role):
                st.write(message)
            if i < len(st.session_state.conversation) - 1:
                st.divider()

    # 用户输入问题
    question = st.chat_input("请输入您的问题...")

    if question:
        # 添加到对话历史
        st.session_state.conversation.append(f"用户: {question}")

        with st.chat_message(name="🕵️‍♂️ 用户"):
            st.write(question)

        # 模拟检索和生成答案的过程
        with st.spinner("正在思考并生成答案..."):
            time.sleep(1)  # 模拟处理时间

            # 简化的检索过程（实际应用中会使用embedding等高级技术）
            matching_chunks = []
            for item in st.session_state.knowledge_base:
                if any(word in item['content'] for word in question.split()[:3]):
                    matching_chunks.append(item)

            # 模拟生成答案
            if matching_chunks:
                answer = f"根据知识库中的内容为您解答：\n\n"
                answer += matching_chunks[0]['content'][:300] + "\n\n"
                answer += "（此为模拟回答，实际系统会结合上下文生成更自然的回答）"

                # 显示来源
                answer += f"\n\n来源: {matching_chunks[0]['source']}"
            else:
                answer = "在知识库中未找到相关信息。请尝试重新表述您的问题或上传更多相关文档。"

            # 添加到对话历史
            st.session_state.conversation.append(f"系统: {answer}")

            # 显示答案
            with st.chat_message(name="🤖 系统"):
                st.write(answer)


# 主界面
def main():
    st.set_page_config(
        page_title="智能问答系统",
        page_icon="🤖",
        layout="wide"
    )

    st.title("基于RAG的智能问答系统")
    st.caption("支持文档处理与智能问答功能 | 项目实现方案")

    init_session()

    # 创建侧边栏导航
    with st.sidebar:
        st.header("导航")
        page = st.radio("选择功能", ["知识库管理", "智能问答"])
        st.divider()

        st.info("""
        **系统功能：**
        1. 知识库构建与管理
        2. 文档解析与处理
        3. 自然语言问答
        4. 对话上下文记录
        """)

    # 根据选择显示对应页面
    if page == "知识库管理":
        knowledge_base_section()
    else:
        qa_interface()

    # 调试信息
    with st.expander("调试信息"):
        st.json({
            "文件上传数": len(st.session_state.uploaded_files),
            "知识片段数": len(st.session_state.knowledge_base),
            "对话轮次": len(st.session_state.conversation) // 2
        })


if __name__ == "__main__":
    main()