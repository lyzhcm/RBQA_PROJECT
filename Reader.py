import os
import json
import pandas as pd
import PyPDF2

class Reader:
    def read(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.txt', '.csv', '.md', '.log']:
            return self._read_text(file_path)
        elif ext in ['.json']:
            return self._read_json(file_path)
        elif ext in ['.xlsx', '.xls']:
            return self._read_excel(file_path)
        elif ext in ['.pdf']:
            return self._read_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _read_text(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _read_excel(self, file_path):
        return pd.read_excel(file_path)

    def _read_pdf(self, file_path):
        try:
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    if hasattr(page, 'extract_text'):
                        text += page.extract_text() or ""
                    else:
                        raise ValueError("The 'extract_text' method is not available in this version of PyPDF2.")
            return text
        except ImportError:
            raise ImportError("Please install PyPDF2 to read PDF files.")
        except Exception as e:
            raise ValueError(f"An error occurred while reading the PDF file: {e}")

# 示例用法
# reader = Reader()
# content = reader.read('example.txt')
# print(content)