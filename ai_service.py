import openai
import streamlit as st
from config import API_KEY, API_BASE_URL, DEEPSEEK_MODEL

openai.api_key = API_KEY
openai.base_url = API_BASE_URL

def ask_ai(prompt, model=DEEPSEEK_MODEL):
    """调用DeepSeek AI接口"""
    # 消息历史应由调用者（UI层）管理。
    # 此函数现在接收一个完整的提示字符串。
    messages = [
        {
            "role": "system",
            "content": "你是一个帮助进行文献管理的AI问答系统。根据提供的上下文，用精炼而科学的语言回答问题"
        },
        {"role": "user", "content": prompt}
    ]
    try:
        completion = openai.chat.completions.create(
            model=model,
            messages=messages,
        )
        ai_reply = completion.choices[0].message.content.strip()
        return ai_reply
    except Exception as e:
        st.error(f"AI接口调用失败: {str(e)}")
        return "无法获取AI回答，请检查API配置"
        st.error(f"AI接口调用失败: {str(e)}")
        return "无法获取AI回答，请检查API配置"

