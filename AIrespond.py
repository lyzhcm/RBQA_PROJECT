import openai

openai.api_key = "sk-g40Ua40lLiQhMcEN1b710a5d63E14bD89921Ed47D8B371Fb"
openai.base_url = "https://api.gpt.ge/v1/"

def ask_ai(question, model="deepseek-chat"):
    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个帮助进行文献管理的AI问答系统。现在你需要根据我接下来给出的关键词，检索相关信息，用精炼而科学的语言回答问题"},
            {"role": "user", "content": question},
        ],
    )
    return completion.choices[0].message.content.strip()

if __name__ == "__main__":
    user_question = input("请输入你的问题：")
    answer = ask_ai(user_question)
    print("AI回答：", answer)