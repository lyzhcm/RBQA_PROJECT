#file_registry
import json
from pathlib import Path
from datetime import datetime
from config import FILE_REGISTRY_DB

class FileRegistry:
    @staticmethod
    def load():
        """加载文件注册表"""
        try:
            if Path(FILE_REGISTRY_DB).exists():
                with open(FILE_REGISTRY_DB, 'r') as f:
                    return json.load(f)
            return {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def save(registry):
        """保存文件注册表"""
        with open(FILE_REGISTRY_DB, 'w') as f:
            json.dump(registry, f, indent=2)

    @staticmethod
    def add_file(file_id, filename, filepath):
        """注册新文件"""
        registry = FileRegistry.load()
        registry[file_id] = {
            "filename": filename,
            "filepath": str(Path(filepath).absolute()),
            "timestamp": datetime.now().isoformat()
        }
        FileRegistry.save(registry)

    @staticmethod
    def remove_file(file_id):
        """移除文件注册"""
        registry = FileRegistry.load()
        if file_id in registry:
            del registry[file_id]
            FileRegistry.save(registry)
            return True
        return False