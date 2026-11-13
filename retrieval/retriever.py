
class DocumentRetriever:

    def __init__(self, vector_store):

        # Access the existing ChromaDB instance
        self.db = vector_store.vector_store

    def similarity_search(self, query, k=4, metadata_filter=None):

        return self.db.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter
        )

    def mmr_search(
        self,
        query,
        k=4,
        fetch_k=20,
        lambda_mult=0.5,
        metadata_filter=None
    ):

        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")

        return self.db.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=metadata_filter
        )
