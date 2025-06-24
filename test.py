import streamlit as st
from Reader import Reader

# 创建 Streamlit 应用标题
st.title("PDF 文件内容展示")

# 创建 Reader 类的实例
reader = Reader()

# 文件路径输入
file_path = st.text_input("请输入 PDF 文件路径:", "H:\\RBQA_PROJECT\\RBQA_PROJECT\\sheet.pdf")

# 读取并展示 PDF 内容
if st.button("读取文件"):
    try:
        content = reader.read(file_path)
        st.text_area("文件内容:", content, height=300)
    except Exception as e:
        st.error(f"发生错误: {e}")