from embedding.embedding_model import EmbeddingModel
from vectorstore.chroma_store import ChromaVectorStore


# Initialize embedding service
embedding_service = EmbeddingModel()

# Connect to existing ChromaDB collection
vector_db = ChromaVectorStore(
    embedding_model=embedding_service.model,
    persist_directory="./data/chroma_db",
    collection_name="document_intelligence"
)

# User query
query = "How does supervised learning work?"

# Retrieve documents with similarity distances
results = vector_db.similarity_search_with_score(
    query=query,
    k=4
)

print("\n--- VECTOR SEARCH RESULTS ---")

for index, (document, distance) in enumerate(results):

    print(f"\nResult {index + 1}")
    print("Distance:", distance)
    print("Content:", document.page_content[:300])
    print("Metadata:", document.metadata)