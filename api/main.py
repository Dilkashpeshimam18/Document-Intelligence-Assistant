from ingestion.document_loader import DocumentIngestion
from ingestion.document_cleaner import DocumentCleaner
from ingestion.metadata_extractor import MetadataExtractor
from chunking.text_splitter import (FixedSizedChunker, RecursiveChunker, TokenBasedChunker) 
from chunking.semantic_chunker import SemanticChunkerService
from chunking.parent_child_chunker import ParentChildChunker
from chunking.chunking_evaluator import ChunkingEvaluator
from embedding.embedding_model import EmbeddingModel
from vectorstore.chroma_store import ChromaVectorStore

ingestion=DocumentIngestion(
    data_path='data/raw'
)

chunker=FixedSizedChunker(
    chunk_size=500,
    chunk_overlap=50
)

recursive_chunker = RecursiveChunker(
    chunk_size=500,
    chunk_overlap=50
)

# Initialize embedding service
embedding_service = EmbeddingModel()

# Initialize ChromaDB
vector_db = ChromaVectorStore(
    embedding_model=embedding_service.model,
    persist_directory="./data/chroma_db",
    collection_name="document_intelligence"
)

evaluator = ChunkingEvaluator()

documents=ingestion.load_all_documents()

print("Total document pages/files:", len(documents))

cleaner= DocumentCleaner()

cleanedDocument=cleaner.clean_documents(
    documents
)

print("Cleaned documents:", len(cleanedDocument))

extractor=MetadataExtractor()

processed_documents=extractor.extract_metadata(
    cleanedDocument
)

chunks=chunker.split_documents(processed_documents)
print("Processed documents:", len(processed_documents))

print("\n========== CHUNKING RESULTS ==========")

print("Original documents:", len(processed_documents))
print("Generated chunks:", len(chunks))

recursive_chunks = recursive_chunker.split_documents(
    processed_documents
)

print("\n========== RECURSIVE CHUNKING RESULTS ==========")

print("Original documents:", len(processed_documents))

print("Generated chunks:", len(recursive_chunks))


# Display sample chunks
for i, chunk in enumerate(chunks[:3]):

    print(f"\n========== CHUNK {i + 1} ==========")

    print("Content:")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)

    print("Character count:", len(chunk.page_content))



# STEP: Token-Based Chunking

token_chunker = TokenBasedChunker(
    chunk_size=300,
    chunk_overlap=30,
    encoding_name="cl100k_base"
)

token_chunks = token_chunker.split_documents(
    processed_documents
)

print("\n========== TOKEN CHUNKING RESULTS ==========")

print("Original documents:", len(processed_documents))

print("Generated token chunks:", len(token_chunks))


# Display sample chunks

for i, chunk in enumerate(token_chunks[:3]):

    print(f"\n========== TOKEN CHUNK {i + 1} ==========")

    print("Content:")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)

semantic_chunker = SemanticChunkerService()

semantic_chunks = semantic_chunker.split_documents(documents)

print("\n--- Semantic Chunking ---")
print("Total semantic chunks:", len(semantic_chunks))

for i, chunk in enumerate(semantic_chunks[:3]):
    print(f"\nChunk {i + 1}:")
    print(chunk.page_content[:500])
    print("Metadata:", chunk.metadata)


parent_child_chunker = ParentChildChunker(
    parent_chunk_size=2000,
    parent_chunk_overlap=200,
    child_chunk_size=400,
    child_chunk_overlap=50
)

# Replace documents with your metadata-enriched document variable
result = parent_child_chunker.split_documents(documents)

parents = result["parents"]
children = result["children"]
parent_store = result["parent_store"]

print("\n--- Parent-Child Chunking ---")
print("Total parent chunks:", len(parents))
print("Total child chunks:", len(children))

if children:
    print("\nSample child:")
    print(children[0].page_content[:300])
    print("Parent ID:", children[0].metadata["parent_id"])

    retrieved_parents = parent_child_chunker.retrieve_parents(
        [children[0]],
        parent_store
    )

    print("\nRetrieved parent:")
    print(retrieved_parents[0].page_content[:500])

# Evaluate each strategy
results = [
    evaluator.evaluate(chunks, "Fixed-size"),
    evaluator.evaluate(recursive_chunks, "Recursive"),
    evaluator.evaluate(token_chunks, "Token-based")
]

evaluator.compare(results)


# Index our document chunks
ids = vector_db.add_documents(recursive_chunks)

print("\n--- ChromaDB Indexing ---")
print("Indexed chunks:", len(ids))

# Search using a natural-language question
query = "What is machine learning?"

results = vector_db.similarity_search(
    query=query,
    k=4
)

print("\n--- Retrieved Documents ---")

for i, document in enumerate(results):

    print(f"\nResult {i + 1}:")
    print(document.page_content[:500])
    print("Metadata:", document.metadata)