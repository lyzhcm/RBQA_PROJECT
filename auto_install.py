import subprocess
import sys
import os

REQUIRED_PACKAGES = [
    # UI框架
    "streamlit",
    
    # 数据处理
    "pandas",
    "numpy",
    
    # HTTP请求
    "requests",
    
    # AI和机器学习
    "openai",
    "sentence-transformers",
    "transformers",
    "torch",
    
    # LangChain生态系统
    "langchain",
    "langchain-community",
    
    # 向量数据库
    "chromadb",
    
    # 自然语言处理
    "nltk",
    "tokenizers",
    
    # 文档解析
    "PyPDF2",
    "python-docx",
    "python-pptx",  # 保持包名不变，但会修复导入
    "beautifulsoup4",
    "openpyxl",
    
    # Hugging Face相关
    "huggingface-hub",
    
    # 系统和进程管理
    "psutil",
    
    # 文件处理和安全
    "werkzeug",
    "pathlib2",
    
    # 其他实用工具
    "python-multipart",
    "typing-extensions",
    
    # 关键兼容性锁定
    "protobuf==4.25.3",
    
    # 打包工具
    "pyinstaller"
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-i", "https://pypi.tuna.tsinghua.edu.cn/simple", package])

def create_executable():
    """创建可执行文件"""
    print("\n正在创建可执行文件...")
    
    # 1. 创建一个包装脚本来运行streamlit
    wrapper_script_content = '''
import streamlit.web.cli as stcli
import sys
import os
import io

if __name__ == '__main__':
    # 修复PyInstaller环境下的stdin问题
    if not hasattr(sys, 'stdin') or sys.stdin is None:
        sys.stdin = io.StringIO()
    
    # 获取main.py的路径
    # sys._MEIPASS 是PyInstaller在运行时创建的临时文件夹路径
    if hasattr(sys, '_MEIPASS'):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    main_py_path = os.path.join(base_path, 'main.py')
    
    print(f"正在启动 Streamlit 应用...")
    print(f"主文件路径: {main_py_path}")
    print(f"服务器将在 http://localhost:8501 启动")
    print("=" * 50)
    
    # 设置streamlit运行所需的参数
    # 移除端口设置以避免开发模式冲突
    sys.argv = [
        "streamlit", 
        "run", 
        main_py_path, 
        "--server.headless=true", 
        "--server.enableCORS=false",
        "--global.developmentMode=false"
    ]
    
    try:
        stcli.main()
    except Exception as e:
        print(f"启动Streamlit失败: {e}")
        print("请检查main.py文件是否存在且正确")
        # 在打包环境下不使用input()
        if hasattr(sys, '_MEIPASS'):
            import time
            print("程序将在5秒后退出...")
            time.sleep(5)
        else:
            input("按Enter键退出...")
'''
    
    wrapper_script_path = "run_streamlit.py"
    with open(wrapper_script_path, "w", encoding="utf-8") as f:
        f.write(wrapper_script_content)

    # 2. 收集所有需要打包的.py文件
    python_files = []
    for filename in os.listdir('.'):
        if filename.endswith('.py') and filename != wrapper_script_path:
            python_files.append(f"--add-data={filename};.")

    # 3. 运行PyInstaller
    try:
        command = [
            sys.executable, "-m", "PyInstaller",
            "--onefile",  # 打包成单个exe文件
            # 移除 --windowed 参数以显示命令行窗口
            "--name=RBQA_APP",  # 指定输出的exe文件名
            "--add-data=requirements.txt;.",  # 添加依赖文件
            "--hidden-import=streamlit",
            "--hidden-import=streamlit.web.cli",
            "--hidden-import=langchain",
            "--hidden-import=chromadb",
            "--hidden-import=sentence_transformers",
            "--hidden-import=pptx",  # 添加pptx隐藏导入
            "--hidden-import=docx", 
            "--hidden-import=PyPDF2",
            "--collect-all=streamlit",
            "--collect-all=altair",
            *python_files,  # 添加所有Python文件
            wrapper_script_path
        ]
        
        print(f"正在执行命令...")
        print(" ".join(command))
        subprocess.check_call(command)
        print("\n✅ 可执行文件 'dist/RBQA_APP.exe' 创建成功！")
        print("\n📌 使用说明：")
        print("1. 运行 dist/RBQA_APP.exe")
        print("2. 程序会显示命令行窗口并启动Streamlit服务器")
        print("3. 浏览器会自动打开应用界面")
        print("4. 如果浏览器未自动打开，请访问 http://localhost:8501")
        print("5. 关闭命令行窗口会停止应用")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 创建可执行文件失败: {e}")
        print("请确保已正确安装所有依赖包")
    except FileNotFoundError:
        print("\n❌ 错误: PyInstaller未找到")
        print("请确保已安装PyInstaller: pip install pyinstaller")
    finally:
        # 4. 清理临时文件
        cleanup_files = [wrapper_script_path, "RBQA_APP.spec"]
        for file in cleanup_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"已清理临时文件: {file}")
                except Exception as e:
                    print(f"清理文件失败 {file}: {e}")

if __name__ == "__main__":
    print("🚀 开始安装依赖包...")
    for pkg in REQUIRED_PACKAGES:
        try:
            # 处理包名中的版本号和特殊包名映射
            pkg_name = pkg.split("==")[0]
            # 特殊包名映射
            import_name_mapping = {
                "python-docx": "docx",
                "python-pptx": "pptx", 
                "beautifulsoup4": "bs4",
                "python-multipart": "multipart",
                "sentence-transformers": "sentence_transformers",
                "langchain-community": "langchain_community",
                "huggingface-hub": "huggingface_hub"
            }
            
            # 获取实际的导入名称
            import_name = import_name_mapping.get(pkg_name, pkg_name.replace("-", "_"))
            __import__(import_name)
            print(f"✅ 已安装: {pkg}")
        except ImportError:
            print(f"⏳ 正在安装: {pkg}")
            install(pkg)
    
    print("\n✅ 所有依赖已安装完毕！")

    # 检查main.py是否存在
    if os.path.exists("main.py"):
        print("\n📦 创建可执行文件选项：")
        choice = input("是否要创建可执行文件 (exe)? [y/N]: ").lower().strip()
        if choice in ['y', 'yes']:
            create_executable()
        else:
            print("\n🏃 你可以通过以下命令运行程序：")
            print("streamlit run main.py")
    else:
        print("\n⚠️  未找到 main.py 文件")
        print("请确保 main.py 文件在当前目录下")
        print("跳过创建可执行文件步骤")
    
    print("\n🎉 安装程序执行完毕！")
    input("按Enter键退出...")