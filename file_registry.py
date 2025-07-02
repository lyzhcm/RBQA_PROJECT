#file_registry
from datetime import datetime

# 用于内存注册表
_MEM_REGISTRY = {}

class FileRegistry:
    @staticmethod
    def load():
        """加载文件注册表（仅内存）"""
        return _MEM_REGISTRY

    @staticmethod
    def save(registry):
        """保存文件注册表（仅内存，无操作）"""
        global _MEM_REGISTRY
        _MEM_REGISTRY = registry

    @staticmethod
    def add_file(file_id, filename, filepath):
        """注册新文件（仅内存）"""
        registry = FileRegistry.load()
        registry[file_id] = {
            "filename": filename,
            "filepath": filepath,
            "timestamp": datetime.now().isoformat()
        }
        FileRegistry.save(registry)

    @staticmethod
    def remove_file(file_id):
        """移除文件注册（仅内存）"""
        registry = FileRegistry.load()
        if file_id in registry:
            del registry[file_id]
            FileRegistry.save(registry)
            return True
        return False