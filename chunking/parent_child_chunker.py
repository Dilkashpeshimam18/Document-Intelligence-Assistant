
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ParentChildChunker:

    def __init__(
        self,
        parent_chunk_size=2000,
        parent_chunk_overlap=200,
        child_chunk_size=400,
        child_chunk_overlap=50
    ):

        # Split documents into larger parent sections
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            add_start_index=True
        )

        # Split each parent into smaller child sections
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            add_start_index=True
        )

    def split_documents(self, documents):

        parent_documents = []
        child_documents = []
        parent_store = {}

        for document in documents:

            # Step 1: Generate parent chunks
            parents = self.parent_splitter.split_documents(
                [document]
            )

            for parent in parents:

                # Step 2: Give every parent a unique ID
                parent_id = str(uuid4())

                parent.metadata["parent_id"] = parent_id

                parent_documents.append(parent)

                # Step 3: Store parent for later retrieval
                parent_store[parent_id] = parent

                # Step 4: Generate children from this parent
                children = self.child_splitter.split_documents(
                    [parent]
                )

                for child_index, child in enumerate(children):

                    child.metadata["parent_id"] = parent_id
                    child.metadata["child_index"] = child_index

                    child_documents.append(child)

        return {
            "parents": parent_documents,
            "children": child_documents,
            "parent_store": parent_store
        }

    @staticmethod
    def retrieve_parents(retrieved_children, parent_store):

        # Find parent IDs from retrieved child chunks
        parent_ids = {
            child.metadata["parent_id"]
            for child in retrieved_children
            if child.metadata.get("parent_id") in parent_store
        }

        # Return the corresponding parent documents
        return [
            parent_store[parent_id]
            for parent_id in parent_ids
        ]
