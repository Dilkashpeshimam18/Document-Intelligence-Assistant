
from pathlib import Path
from hashlib import sha256

from langchain_chroma import Chroma


class ChromaVectorStore:

    def __init__(
        self,
        embedding_model,
        persist_directory="./data/chroma_db",
        collection_name="document_intelligence"
    ):

        self.persist_directory = str(
            Path(persist_directory).resolve()
        )

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True
        )

        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=self.persist_directory
        )

    def _generate_ids(self, documents):

        ids = []

        for document in documents:

            identity = (
                str(document.metadata.get("document_id", ""))
                + str(document.metadata.get("start_index", ""))
                + document.page_content
            )

            chunk_id = sha256(
                identity.encode("utf-8")
            ).hexdigest()

            ids.append(chunk_id)

        return ids

    def add_documents(self, documents):

        if not documents:
            return []

        ids = self._generate_ids(documents)

        self.vector_store.add_documents(
            documents=documents,
            ids=ids
        )

        return ids

    def similarity_search(self, query, k=4):

        return self.vector_store.similarity_search(
            query=query,
            k=k
        )

    def similarity_search_with_score(self, query, k=4):

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k
        )

    def delete_documents(self, ids):

        if ids:
            self.vector_store.delete(ids=ids)
