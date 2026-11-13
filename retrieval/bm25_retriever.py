
from langchain_community.retrievers import BM25Retriever


class BM25DocumentRetriever:

    def __init__(self, documents, k=4):

        if not documents:
            raise ValueError("Documents cannot be empty")

        # Build the BM25 index from document chunks
        self.retriever = BM25Retriever.from_documents(
            documents,
            k=k
        )

    def search(self, query):

        return self.retriever.invoke(query)
