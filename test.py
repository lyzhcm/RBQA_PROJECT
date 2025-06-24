from Reader import Reader

# 创建 Reader 类的实例
reader = Reader()

# 调用 read 方法读取文件内容
file_path = 'example.pdf'  # 替换为你的文件路径
content = reader.read(file_path)

# 打印读取的内容
print(content)