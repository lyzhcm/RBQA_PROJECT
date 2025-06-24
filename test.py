import streamlit as st
from Reader import Reader
import os
import tempfile

# --- 核心功能函数 (升级版) ---

def get_answer_from_context(knowledge_base, question):
    """
    升级版问答函数，支持在多文件知识库中检索，并根据相关性排序。
    """
    # 1. 准备数据：将所有文件的内容和来源整合在一起
    all_sentences_with_source = []
    for filename, content in knowledge_base.items():
        # 按行分割，并为每句话附上来源文件名
        sentences = content.split('\n')
        for sentence in sentences:
            if sentence.strip(): # 忽略空行
                all_sentences_with_source.append({"sentence": sentence, "source": filename})

    # 2. 检索：基于关键字匹配和评分
    question_words = set(question.lower().split())
    if not question_words:
        return "请输入一个有效的问题。"

    relevant_sentences = []
    for item in all_sentences_with_source:
        sentence_lower = item["sentence"].lower()
        # 计算得分：句子中包含了多少个不同的问题关键字
        score = sum(1 for word in question_words if word in sentence_lower)
        if score > 0:
            relevant_sentences.append({
                "sentence": item["sentence"],
                "source": item["source"],
                "score": score
            })

    # 3. 排序和构造答案
    if relevant_sentences:
        # 按得分从高到低排序
        sorted_sentences = sorted(relevant_sentences, key=lambda x: x['score'], reverse=True)
        
        # 构造答案，最多引用前5条最相关的信息
        top_n = 5
        answer = "根据您提供的文档，我找到了以下相关信息：\n\n"
        
        # 使用集合确保不重复显示完全相同的句子
        unique_sents = set()
        count = 0
        for item in sorted_sentences:
            if item['sentence'] not in unique_sents:
                answer += f"- **来源: *{item['source']}***\n"
                answer += f"  > {item['sentence']}\n\n"
                unique_sents.add(item['sentence'])
                count += 1
                if count >= top_n:
                    break
        return answer
    else:
        return "抱歉，在您上传的文档中未能找到与您问题相关的信息。请尝试换个问法。"

# --- Streamlit 应用界面 ---

# 1. 页面配置
st.set_page_config(page_title="知识库问答助手", page_icon="📚", layout="wide")

# 2. 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = {} # 从单个内容变为字典形式的知识库

# 3. 侧边栏
with st.sidebar:
    st.title("🧠 知识库管理")
    st.write("请上传您的文档来构建知识库。")

    uploaded_files = st.file_uploader(
        "可上传多个文件",
        type=['txt', 'csv', 'md', 'log', 'json', 'xlsx', 'xls', 'pdf'],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # 如果文件未被处理过，则进行处理
            if uploaded_file.name not in st.session_state.knowledge_base:
                with st.spinner(f"正在解析文件: {uploaded_file.name}..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        reader = Reader()
                        content = reader.read(tmp_file_path)
                        st.session_state.knowledge_base[uploaded_file.name] = content
                        os.unlink(tmp_file_path)
                        st.success(f"文件 '{uploaded_file.name}' 已添加至知识库！")
                    except Exception as e:
                        st.error(f"解析 '{uploaded_file.name}' 时出错: {e}")

    # 显示当前知识库中的文件列表
    if st.session_state.knowledge_base:
        st.divider()
        st.subheader("当前知识库文件:")
        for filename in st.session_state.knowledge_base.keys():
            st.markdown(f"- 📄 {filename}")
        
        # 提供清空知识库的按钮
        if st.button("清空知识库和对话", type="primary"):
            st.session_state.knowledge_base = {}
            st.session_state.messages = []
            st.rerun() # 重新运行应用以刷新界面

# 4. 主聊天界面
st.title("智能文档问答助手 💬")

# 显示欢迎信息或聊天记录
if not st.session_state.knowledge_base:
    st.info("请在左侧侧边栏上传文档来构建您的知识库。")
else:
    # 显示历史对话
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 接收用户输入
    if prompt := st.chat_input("请就知识库中的内容提问..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("正在知识库中检索答案..."):
                response = get_answer_from_context(st.session_state.knowledge_base, prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})