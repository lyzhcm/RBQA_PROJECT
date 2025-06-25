#knowledge_base_manager
import re
import streamlit as st
from datetime import datetime
from file_parser import parse_file, generate_file_id,save_uploaded_file
import vector_store as db_op
import os
from file_registry import FileRegistry
from pathlib import Path
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

# 新增文件到知识库
def add_file_to_knowledge_base(file):
    """Processes an uploaded file and adds it to the knowledge base and vector store."""
    # 判断 file 是 UploadedFile 还是 Path
    if hasattr(file, "getvalue"):
        file_id = generate_file_id(file.getvalue())
    elif isinstance(file, Path):
        with open(file, "rb") as f:
            file_id = generate_file_id(f.read())
    else:
        raise ValueError("不支持的文件类型")

    # 检查文件是否已持久化注册
    registry = FileRegistry.load()
    if file_id in registry:
        file_info = registry[file_id]
        file_path = Path(file_info["filepath"])
        with open(file_path, "rb") as f:
            content = parse_file(f)
    else:
        content = parse_file(file)
        save_uploaded_file(file, file_id)  # 保存到持久化存储

    existing_file = next((f for f in st.session_state.uploaded_files if f['id'] == file_id), None)

    if existing_file:
        return

    with st.spinner(f"解析文件: {file.name}..."):
        content = parse_file(file)
        if not content:
            return

        st.session_state.uploaded_files.append({
            "id": file_id,
            "name": file.name,
            "type": file.type,
            "content": content,
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tags": ["新上传"]
        })

        chunks = st.session_state.text_splitter.split_text(content)
        # 过滤掉空chunk
        filtered = [(c, {
            "source": file.name,
            "source_id": file_id,
            "type": file.type.split("/")[-1],
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }) for c in chunks if c and isinstance(c, str) and c.strip() != ""]
        if not filtered:
            st.warning("未提取到有效文本片段，未入库。")
            return
        chunks, metadatas = zip(*filtered)
        chunks = list(chunks)
        metadatas = list(metadatas)

        # 确保所有metadata字段为字符串
        for m in metadatas:
            for k, v in m.items():
                if v is None:
                    m[k] = ""
                elif not isinstance(v, (str, int, float)):
                    m[k] = str(v)

        db_op.add_texts_to_db(texts=chunks, metadatas=metadatas)

        for chunk in chunks:
            st.session_state.knowledge_base.append({
                "source": file.name,
                "source_id": file_id,
                "content": chunk,
                "type": file.type.split("/")[-1]
            })

# 删除文件处理
def delete_file(file_id):
    """删除文件时同时清理持久化存储"""
    file_to_delete = next((f for f in st.session_state.uploaded_files if f['id'] == file_id), None)
    if file_to_delete:
        try:
            # 删除物理文件
            if os.path.exists(file_to_delete['local_path']):
                os.remove(file_to_delete['local_path'])
            
            # 移除注册信息
            FileRegistry.remove_file(file_id)
            
            # 更新会话状态
            st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f['id'] != file_id]
            st.session_state.deleted_files.append({
                **file_to_delete,
                'deleted_time': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            st.error(f"文件删除失败: {str(e)}")
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
        
        # 重新分块并添加到知识库和向量数据库
        content = file_to_restore['content']
        chunks = st.session_state.text_splitter.split_text(content)
        
        metadatas = [{
            "source": file_to_restore['name'],
            "source_id": file_id,
            "type": file_to_restore['type'].split("/")[-1],
            "upload_time": file_to_restore['upload_time']
        } for _ in chunks]
        db_op.add_texts_to_db(texts=chunks, metadatas=metadatas)

        for chunk in chunks:
            st.session_state.knowledge_base.append({
                "source": file_to_restore['name'],
                "source_id": file_id,
                "content": chunk,
                "type": file_to_restore['type'].split("/")[-1]
            })
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