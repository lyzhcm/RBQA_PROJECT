import openai
import streamlit as st

openai.api_key = "sk-g40Ua40lLiQhMcEN1b710a5d63E14bD89921Ed47D8B371Fb"
openai.base_url = "https://api.gpt.ge/v1/"

def ask_ai(question, model="deepseek-chat"):
    """调用DeepSeek AI接口"""
    try:
        completion = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个帮助进行文献管理的AI问答系统。根据提供的上下文，用精炼而科学的语言回答问题"
                },
                {"role": "user", "content": question},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"AI接口调用失败: {str(e)}")
        return "无法获取AI回答，请检查API配置"

