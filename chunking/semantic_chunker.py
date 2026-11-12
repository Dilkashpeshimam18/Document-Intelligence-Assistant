from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


class SemanticChunkerService:

    def __init__(self):

        # Convert sentences into numerical vectors
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Create semantic chunker
        self.splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )

    def split_documents(self, documents):

        chunks = self.splitter.split_documents(documents)

        return chunks
