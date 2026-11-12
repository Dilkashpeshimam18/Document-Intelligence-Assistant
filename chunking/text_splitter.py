from langchain_text_splitters import (CharacterTextSplitter, RecursiveCharacterTextSplitter,TokenTextSplitter)

class FixedSizedChunker:
    def __init__(self,chunk_size=500,chunk_overlap=50):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap

        self.splitter=CharacterTextSplitter(
            separator='',
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    def split_documents(self,documents):
        chunks=self.splitter.split_documents(documents)
        return chunks


class RecursiveChunker:

    def __init__(
        self,
        chunk_size=500,
        chunk_overlap=50
    ):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                " ",
                ""
            ],
            add_start_index=True
        )

    def split_documents(self, documents):

        chunks = self.splitter.split_documents(documents)

        return chunks

     

class TokenBasedChunker:

    def __init__(
        self,
        chunk_size=300,
        chunk_overlap=30,
        encoding_name="cl100k_base"
    ):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize token-based splitter
        self.splitter = TokenTextSplitter(
            encoding_name=encoding_name,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_documents(self, documents):

        chunks = self.splitter.split_documents(documents)

        return chunks
