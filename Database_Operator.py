import re
import streamlit as st
from datetime import datetime

# 语义分析函数
def semantic_analysis(question):
    # 1. 意图识别
    intent = "信息查询"  # 默认意图

    if any(word in question.lower() for word in ["如何", "怎样", "步骤"]):
        intent = "操作指导"
    elif any(word in question.lower() for word in ["为什么", "原因", "为何"]):
        intent = "原因解释"
    elif any(word in question.lower() for word in ["比较", "对比", "vs"]):
        intent = "比较分析"
    elif any(word in question.lower() for word in ["推荐", "建议", "应该"]):
        intent = "推荐建议"

    # 2. 关键实体提取（简化版）
    entities = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}', question)

    # 3. 生成语义向量
    model = st.session_state.embedding_model
    embedding = model.encode([question])[0] if model else None

    return {
        "intent": intent,
        "entities": entities,
        "embedding": embedding
    }

# 删除文件处理
def delete_file(file_id):
    # 从已上传文件中删除
    file_to_delete = next((f for f in st.session_state.uploaded_files if f['id'] == file_id), None)
    if file_to_delete:
        st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f['id'] != file_id]
        # 添加到删除的文件列表（用于恢复）
        st.session_state.deleted_files.append({
            **file_to_delete,
            'deleted_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        # 从知识库中删除相关片段
        st.session_state.knowledge_base = [kb for kb in st.session_state.knowledge_base if kb['source_id'] != file_id]
        # 更新向量存储
        update_vector_store()
        return True
    return False

# 恢复已删除文件
def restore_file(file_id):
    # 从删除列表中恢复
    file_to_restore = next((f for f in st.session_state.deleted_files if f['id'] == file_id), None)
    if file_to_restore:
        st.session_state.uploaded_files.append({
            'id': file_to_restore['id'],
            'name': file_to_restore['name'],
            'type': file_to_restore['type'],
            'content': file_to_restore['content'],
            'upload_time': file_to_restore['upload_time'],
            'tags': file_to_restore.get('tags', [])
        })
        # 从删除列表中移除
        st.session_state.deleted_files = [f for f in st.session_state.deleted_files if f['id'] != file_id]
        # 重新添加到知识库
        if "text_splitter" in st.session_state:
            chunks = st.session_state.text_splitter.split_text(file_to_restore['content'])
        else:
            # 兜底：简单按段落分割
            chunks = file_to_restore['content'].split('\n\n')
        for chunk in chunks:
            st.session_state.knowledge_base.append({
                "source": file_to_restore['name'],
                "source_id": file_to_restore['id'],
                "content": chunk,
                "type": file_to_restore['type'].split("/")[-1]
            })
        # 更新向量存储
        update_vector_store()
        return True
    return False

# 标记文件功能
def toggle_file_tag(file_id, tag):
    for file in st.session_state.uploaded_files:
        if file['id'] == file_id:
            if 'tags' not in file:
                file['tags'] = []
            if tag in file['tags']:
                file['tags'].remove(tag)
            else:
                file['tags'].append(tag)
            return True
    return False

# 更新向量存储
def update_vector_store():
    if st.session_state.embedding_model and st.session_state.knowledge_base:
        model = st.session_state.embedding_model
        contents = [kb["content"] for kb in st.session_state.knowledge_base]
        st.session_state.knowledge_vectors = model.encode(contents)
