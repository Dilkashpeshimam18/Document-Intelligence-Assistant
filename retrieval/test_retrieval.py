
from dotenv import load_dotenv

from embedding.embedding_model import EmbeddingModel
from vectorstore.chroma_store import ChromaVectorStore
from retrieval.retriever import DocumentRetriever
from retrieval.query_transformer import QueryTransformer
from retrieval.multi_query_retriever import MultiQueryRetriever


load_dotenv()

# Initialize existing embedding model
embedding_service = EmbeddingModel()

# Connect to existing ChromaDB
vector_db = ChromaVectorStore(
    embedding_model=embedding_service.model,
    persist_directory="./data/chroma_db",
    collection_name="document_intelligence"
)

# Initialize retrieval components
document_retriever = DocumentRetriever(vector_db)
query_transformer = QueryTransformer()

# Test query rewriting
original_query = "How does it work?"

conversation_history = """
We were discussing supervised learning and labelled datasets.
"""

rewritten_query = query_transformer.rewrite_query(
    query=original_query,
    conversation_history=conversation_history
)

print("\n--- QUERY REWRITING ---")
print("Original:", original_query)
print("Rewritten:", rewritten_query)


# Test multi-query retrieval
multi_retriever = MultiQueryRetriever(
    document_retriever=document_retriever,
    query_transformer=query_transformer
)

query = "Explain supervised learning and its applications."

result = multi_retriever.retrieve(
    query=query,
    k=3,
    num_queries=3
)

print("\n--- GENERATED QUERIES ---")

for search_query in result["queries"]:
    print("-", search_query)

print("\n--- COMBINED RETRIEVAL RESULTS ---")
print("Total unique chunks:", len(result["documents"]))

for index, document in enumerate(result["documents"]):

    print(f"\nResult {index + 1}")
    print(document.page_content[:300])
    print("Metadata:", document.metadata)
