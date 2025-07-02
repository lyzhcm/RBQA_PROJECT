# config.py（仅保留常量）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 强制所有Hugging Face请求走国内镜像
os.environ["HF_HUB_OFFLINE"] = "0"                   # 确保仍允许在线下载（默认值）

# API 配置
# 默认API密钥，可以在应用的UI界面中进行覆盖
API_KEY = "sk-g40Ua40lLiQhMcEN1b710a5d63E14bD89921Ed47D8B371Fb"
API_BASE_URL = "https://api.gpt.ge/v1/"
DEEPSEEK_MODEL = "deepseek-chat"

# 模型配置
EMBEDDING_MODEL_SENTENCE_TRANSFORMER = 'paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_MODEL_LANGCHAIN = "GanymedeNil/text2vec-large-chinese"

# 数据库配置
CHROMA_DB_PATH = "./chroma_db"
FILE_REGISTRY_DB = "./file_registry.json"  # 确保此路径不依赖其他模块

# 文本分割配置
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 持久化存储配置（仅定义路径，不执行操作）
PERSISTENT_UPLOAD_FOLDER = "./persistent_uploads"