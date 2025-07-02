import subprocess
import sys

REQUIRED_PACKAGES = [
    "streamlit",
    "pandas",
    "requests",
    "sentence-transformers",
    "langchain",
    "langchain-community",
    "chromadb",
    "beautifulsoup4",
    "python-docx",
    "nltk",
    "openpyxl",
    "psutil"
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", package])

if __name__ == "__main__":
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"已安装: {pkg}")
        except ImportError:
            print(f"正在安装: {pkg}")
            install(pkg)
    print("所有依赖已安装完毕。你可以运行主程序了。")