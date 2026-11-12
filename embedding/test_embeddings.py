from embedding.embedding_model import EmbeddingModel

embedding_model=EmbeddingModel()

texts = [
    "Machine learning enables computers to learn from data.",
    "Supervised learning uses labelled training data.",
    "The weather is sunny today."
]

# Generate document embeddings
vectors = embedding_model.embed_documents(texts)

print("Number of vectors:", len(vectors))
print("Embedding dimensions:", len(vectors[0]))

# Generate a query embedding
query = "How do computers learn from examples?"

query_vector = embedding_model.embed_query(query)

print("Query vector dimensions:", len(query_vector))
