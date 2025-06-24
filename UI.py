import streamlit as st
import pandas as pd
from datetime import datetime

from Files_Operator import parse_file, generate_file_id, load_embedding_model
from Database_Operator import (
    delete_file,
    restore_file,
    toggle_file_tag,
    semantic_analysis,
    update_vector_store
)
from AI_Respond import ask_ai

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
        # 确保嵌入模型已加载
        if not st.session_state.embedding_model:
            with st.spinner("加载语义模型..."):
                st.session_state.embedding_model = load_embedding_model()

        # 处理新上传的文件
        for file in uploaded_files:
            # 检查是否已上传过相同内容的文件
            file_id = generate_file_id(file.getvalue())
            existing_file = next((f for f in st.session_state.uploaded_files if f['id'] == file_id), None)

            if not existing_file:
                with st.spinner(f"解析文件: {file.name}..."):
                    content = parse_file(file)

                    if content:
                        st.session_state.uploaded_files.append({
                            "id": file_id,
                            "name": file.name,
                            "type": file.type,
                            "content": content,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "tags": ["新上传"]  # 默认标记
                        })

                        # 分割内容为知识片段
                        chunks = st.session_state.text_splitter.split_text(content)
                        
                        # 存储到向量数据库
                        st.session_state.vector_db.add_texts(
                            texts=chunks,
                            metadatas=[{
                                "source": file.name,
                                "source_id": file_id,
                                "type": file.type.split("/")[-1],
                                "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            } for _ in chunks]
                        )

                        # 保留到知识库
                        for i, chunk in enumerate(chunks):
                            st.session_state.knowledge_base.append({
                                "source": file.name,
                                "source_id": file_id,
                                "content": chunk,
                                "type": file.type.split("/")[-1]
                            })

        # 更新向量存储
        update_vector_store()
        st.rerun()

    # 显示上传文件列表
    st.subheader("文档管理")
    st.info("使用以下表格管理您的文档：")

    files_df = pd.DataFrame([
        {
            "ID": f["id"],
            "文件名": f["name"],
            "类型": f["type"],
            "大小": f"{len(f['content']) // 1024} KB",
            "上传时间": f["upload_time"],
            "标记": ", ".join(f.get("tags", [])) if "tags" in f else "",
        }
        for f in st.session_state.uploaded_files
    ])

    # 如果没有任何文件显示提示信息
    if files_df.empty:
        st.info("暂无文档，请上传文档")
    else:
        # 显示数据表格
        st.dataframe(files_df.set_index('ID'), use_container_width=True)

        # 文件管理功能区
        with st.expander("📌 文件操作", expanded=True):
            col1, col2, col3, col4 = st.columns([0.4, 0.2, 0.2, 0.2])

            # 文件选择
            file_ids = list(files_df["ID"])
            selected_file_id = col1.selectbox("选择文件", file_ids, format_func=lambda id: 
                files_df.loc[files_df["ID"] == id, "文件名"].values[0])

            # 标记管理
            tags = ["重要", "待审核", "存档", "参考"]
            selected_tag = col2.selectbox("添加/移除标记", tags)

            # 按钮操作
            if col3.button("应用标记", key="tag_btn", use_container_width=True):
                toggle_file_tag(selected_file_id, selected_tag)
                st.rerun()

            if col4.button("删除文件", key="delete_btn", type="primary", use_container_width=True):
                delete_file(selected_file_id)
                st.rerun()

        # 回收站管理
        if st.session_state.deleted_files:
            with st.expander("🗑️ 回收站管理", expanded=True):
                deleted_df = pd.DataFrame([
                    {
                        "ID": f["id"],
                        "文件名": f["name"],
                        "类型": f["type"],
                        "删除时间": f["deleted_time"],
                    }
                    for f in st.session_state.deleted_files
                ])

                st.dataframe(deleted_df.set_index('ID'), use_container_width=True)

                restore_id = st.selectbox("选择要恢复的文件", deleted_df["ID"], format_func=lambda id: 
                    deleted_df.loc[deleted_df["ID"] == id, "文件名"].values[0])

                if st.button("恢复文件", use_container_width=True):
                    if restore_file(restore_id):
                        st.success(f"文件已恢复")
                        st.rerun()
                    else:
                        st.error("恢复文件失败")

                if st.button("清空回收站", type="primary", use_container_width=True):
                    st.session_state.deleted_files = []
                    st.success("回收站已清空")
                    st.rerun()

        # 查看知识库内容
        with st.expander("🔍 查看知识库内容", expanded=False):
            if not st.session_state.knowledge_base:
                st.info("知识库为空")
            else:
                # 分组显示按文件
                source_files = set(kb['source_id'] for kb in st.session_state.knowledge_base)
                for src_id in list(source_files)[:3]:  # 最多显示前3个文件的内容
                    source_name = next(
                        kb['source'] for kb in st.session_state.knowledge_base if kb['source_id'] == src_id)
                    st.subheader(f"来源: {source_name}")

                    # 显示该文件的前3个片段
                    for kb in [k for k in st.session_state.knowledge_base if k['source_id'] == src_id][:3]:
                        with st.expander(f"知识片段 {kb['content'][:30]}...", expanded=False):
                            st.markdown(kb["content"])
                    st.divider()
                if len(source_files) > 3:
                    st.info(f"已显示3个文件的内容，共{len(source_files)}个文件")

# 问答界面（结合语义理解和DeepSeek）
def qa_interface():
    st.header("💬 智能问答系统")

    # 显示对话历史
    if st.session_state.conversation:
        for msg in st.session_state.conversation:
            role = "user" if msg.startswith("用户:") else "assistant"
            with st.chat_message(role):
                st.write(msg.split(":", 1)[1].strip())

    # 用户提问处理
    if question := st.chat_input("请输入您的问题..."):
        st.session_state.conversation.append(f"用户: {question}")

        # 1. 语义分析处理
        with st.spinner("正在分析问题语义..."):
            semantic_info = semantic_analysis(question)
            intent = semantic_info["intent"]
            entities = semantic_info["entities"]
            question_embedding = semantic_info["embedding"]

        # 2. 向量检索
        docs = st.session_state.vector_db.similarity_search(question, k=3)

        # 3. 构建科学问答提示词
        context = "\n".join([
            f"【文献 {i + 1}】{doc.metadata['source']}\n{doc.page_content}\n"
            for i, doc in enumerate(docs)
        ])

        prompt = f"""根据以下文献内容回答问题：
{context}
问题：{question}
要求：
1. 回答需引用文献（例：【文献1】）
2. 保持学术严谨性
3. 如无相关信息请说明
4. 问题意图：{intent}
5. 关键实体：{', '.join(entities)}
回答："""

        # 4. 调用DeepSeek生成
        with st.chat_message("assistant"):
            with st.spinner("正在生成回答..."):
                answer = ask_ai(prompt)
                st.write(answer)
                st.session_state.conversation.append(f"系统: {answer}")

            # 显示参考文献
            with st.expander("📚 参考文档", expanded=False):
                for i, doc in enumerate(docs, 1):
                    # 兼容无metadata或无source的情况
                    source = getattr(doc, "metadata", {}).get("source", getattr(doc, "source", f"文档{i}"))
                    content = getattr(doc, "page_content", str(doc))[:200] + "..."
                    st.caption(f"【文献{i}】{source}")
                    st.text(content)

            # 显示语义分析详情
            with st.expander("🔍 语义分析详情", expanded=False):
                st.json({
                    "问题意图": intent,
                    "识别实体": entities,
                    "匹配片段数": len(docs),
                    "提示词": prompt[:500] + "..." if len(prompt) > 500 else prompt
                })
