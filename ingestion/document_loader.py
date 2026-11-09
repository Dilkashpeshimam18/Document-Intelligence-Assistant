
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)


class DataIngestion:

    def __init__(self,data_path='data/raw'):
        self.data_path=Path(data_path)

    def load_pdf(self,file_path):
        loader=PyPDFLoader(str(file_path))
        return loader.load()
    
    def load_txt(self, file_path):
        loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )
        return loader.load()

    def load_docx(self, file_path):
        loader = Docx2txtLoader(str(file_path))
        return loader.load()

    def load_all_documents(self):

      documents = []

      for file_path in self.data_path.iterdir():

        if file_path.suffix.lower() == ".pdf":
            docs = self.load_pdf(file_path)

        elif file_path.suffix.lower() == ".txt":
            docs = self.load_txt(file_path)

        elif file_path.suffix.lower() == ".docx":
            docs = self.load_docx(file_path)

        else:
            print(f"Unsupported file: {file_path.name}")
            continue

        documents.extend(docs)

        return documents
