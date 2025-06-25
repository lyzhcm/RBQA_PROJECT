#ui
import streamlit as st
import pandas as pd
from datetime import datetime
from file_registry import FileRegistry
from knowledge_base_manager import (
    delete_file,
    restore_file,
    toggle_file_tag,
    semantic_analysis,
    add_file_to_knowledge_base
)
from ai_service import ask_ai
import vector_store as db_op

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
            add_file_to_knowledge_base(file)
        
        # Rerun to reflect changes immediately
        st.rerun()

    # 显示上传文件列表
    st.subheader("文档管理")
    st.info("使用以下表格管理您的文档：")

    files_df = pd.DataFrame([
        {
            "ID": f["id"],
            "文件名": f["name"],
            "类型": f["type"],
            "大小": f"{len(f['content']) // 1024} KB" if f.get('content') else "未知",
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
                st.warning("知识库为空，请先上传文档。")
                return
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
    st.header("💬💬 智能问答系统")

    # 显示对话历史
    if st.session_state.conversation:
        for msg in st.session_state.conversation:
            role = "user" if msg.startswith("用户:") else "assistant"
            with st.chat_message(role):
                st.write(msg.split(":", 1)[1].strip())

    # 用户提问处理
    if question := st.chat_input("请输入您的问题..."):
        st.session_state.conversation.append(f"用户: {question}")
        
        # 1. 上下文分析 - 检查是否与上一个问题相关
        context_analysis = ""
        if len(st.session_state.conversation) >= 2:
            last_question = st.session_state.conversation[-2]
            if last_question.startswith("用户:"):
                last_question = last_question.split(":", 1)[1].strip()
                
                # 计算当前问题与上一个问题的语义相似度
                model = st.session_state.embedding_model
                last_embedding = model.encode([last_question])[0]
                current_embedding = model.encode([question])[0]
                
                # 使用余弦相似度
                from numpy import dot
                from numpy.linalg import norm
                similarity = dot(last_embedding, current_embedding)/(norm(last_embedding)*norm(current_embedding))
                
                if similarity > 0.7:  # 相似度阈值
                    context_analysis = f"\n注意：这个问题与上一个问题高度相关（相似度{similarity:.2f}），请考虑上下文回答。"
                    # 获取上一个问题的回答
                    last_answer = ""
                    if len(st.session_state.conversation) >= 3:
                        last_answer_msg = st.session_state.conversation[-3]
                        if last_answer_msg.startswith("系统:"):
                            last_answer = last_answer_msg.split(":", 1)[1].strip()
                    
                    context_analysis += f"\n上一个问题: {last_question}\n上一个回答: {last_answer}"

        # 2. 语义分析处理
        with st.spinner("正在分析问题语义..."):
            semantic_info = semantic_analysis(question)
            intent = semantic_info["intent"]
            entities = semantic_info["entities"]
            question_embedding = semantic_info["embedding"]

        # 3. 向量检索
        docs = db_op.search_db(question, k=3)

        # 4. 构建科学问答提示词（增强上下文）
        context = "\n".join([
            f"【文献 {i + 1}】{doc.metadata['source']}\n{doc.page_content}\n"
            for i, doc in enumerate(docs)
        ])

        # 先处理对话历史
        history = '\n'.join([msg for msg in st.session_state.conversation[-6:] if not msg.startswith('系统:')])

        prompt = f"""根据以下文献内容回答问题：
{context}
问题：{question}
{context_analysis}
要求：
1. 回答需引用文献（例：【文献1】）
2. 保持学术严谨性
3. 如无相关信息请说明
4. 问题意图：{intent}
5. 关键实体：{', '.join(entities)}
6. 考虑以下对话历史：
{history}
回答："""

        # 5. 调用DeepSeek生成
        with st.chat_message("assistant"):
            with st.spinner("正在生成回答..."):
                answer = ask_ai(prompt)
                st.write(answer)
                st.session_state.conversation.append(f"系统: {answer}")

            # 显示参考文献
            with st.expander("📚📚 参考文档", expanded=False):
                for i, doc in enumerate(docs, 1):
                    source = getattr(doc, "metadata", {}).get("source", getattr(doc, "source", f"文档{i}"))
                    content = getattr(doc, "page_content", str(doc))[:200] + "..."
                    st.caption(f"【文献{i}】{source}")
                    st.text(content)

            # 显示语义分析详情
            with st.expander("🔍🔍 语义分析详情", expanded=False):
                st.json({
                    "问题意图": intent,
                    "识别实体": entities,
                    "匹配片段数": len(docs),
                    "上下文关联度": f"{similarity:.2f}" if 'similarity' in locals() else "无",
                    "提示词": prompt[:500] + "..." if len(prompt) > 500 else prompt
                })

        if not st.session_state.get("embedding_model"):
            st.error("嵌入模型未加载，请刷新页面或检查模型配置。")
            return

        if not question or not question.strip():
            st.warning("请输入有效的问题。")
            return