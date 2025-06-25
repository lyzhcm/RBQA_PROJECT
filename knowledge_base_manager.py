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
    """处理上传的文件，将其添加到知识库和向量存储中."""
    # 1. 从文件内容生成唯一ID
    file_content = file.getvalue()
    file_id = generate_file_id(file_content)

    # 2. 检查文件是否已在当前会话中处理，防止重复
    if any(f['id'] == file_id for f in st.session_state.uploaded_files):
        return

    # 3. 解析文件内容
    content = parse_file(file)
    if not content:
        st.warning(f"无法从文件 '{file.name}' 中提取文本内容，已跳过。")
        return

    # 4. 持久化保存文件并注册
    save_uploaded_file(file, file_id)

    # 5. 将文件信息添加到会话状态以供UI显示
    st.session_state.uploaded_files.append({
        "id": file_id,
        "name": file.name,
        "type": file.type,
        "content": content,  # 注意：在会话中存储完整内容可能消耗大量内存
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tags": ["新上传"]
    })

    # 6. 对内容进行分块
    chunks = st.session_state.text_splitter.split_text(content)
    
    # 过滤掉空块并准备元数据
    filtered_data = [
        (c, {
            "source": file.name,
            "source_id": file_id,
            "type": file.type.split("/")[-1],
            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }) for c in chunks if c and isinstance(c, str) and c.strip()
    ]

    if not filtered_data:
        st.warning(f"文件 '{file.name}' 未提取到有效文本片段，未存入知识库。")
        return
        
    chunks, metadatas = zip(*filtered_data)
    chunks, metadatas = list(chunks), list(metadatas)

    # 确保所有元数据字段都是ChromaDB接受的类型
    for m in metadatas:
        for k, v in m.items():
            if v is None:
                m[k] = ""
            elif not isinstance(v, (str, int, float, bool)):
                m[k] = str(v)

    # 7. 将块和元数据添加到向量数据库
    db_op.add_texts_to_db(texts=chunks, metadatas=metadatas)

    # 8. 将块信息添加到会话状态知识库（用于UI显示）
    for chunk in chunks:
        st.session_state.knowledge_base.append({
            "source": file.name,
            "source_id": file_id,
            "content": chunk,
            "type": file.type.split("/")[-1]
        })

# 删除文件处理
def delete_file(file_id):
    """从所有地方（会话、持久化存储、向量数据库）删除文件."""
    file_to_delete = next((f for f in st.session_state.uploaded_files if f['id'] == file_id), None)
    if not file_to_delete:
        st.error("尝试删除一个不存在的文件。")
        return False

    try:
        # 从注册表获取文件路径并删除物理文件
        registry = FileRegistry.load()
        file_info = registry.get(file_id)
        if file_info and Path(file_info['filepath']).exists():
            Path(file_info['filepath']).unlink()

        # 从注册表移除记录
        FileRegistry.remove_file(file_id)

        # 从向量数据库删除
        db_op.delete_from_db_by_source_id(file_id)

        # 从会话状态更新
        st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f['id'] != file_id]
        st.session_state.knowledge_base = [kb for kb in st.session_state.knowledge_base if kb['source_id'] != file_id]
        
        # 移动到回收站
        st.session_state.deleted_files.append({
            **file_to_delete,
            'deleted_time': datetime.now().isoformat()
        })
        
        st.toast(f"文件 '{file_to_delete['name']}' 已移至回收站。")
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