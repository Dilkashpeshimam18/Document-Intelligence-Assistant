
class MultiQueryRetriever:

    def __init__(self, document_retriever, query_transformer):

        self.retriever = document_retriever
        self.transformer = query_transformer

    def retrieve(
        self,
        query,
        k=3,
        num_queries=3
    ):

        # Generate alternative queries
        alternative_queries = (
            self.transformer.generate_multi_queries(
                query=query,
                num_queries=num_queries
            )
        )

        # Include original query
        all_queries = list(dict.fromkeys(
            [query] + alternative_queries
        ))

        combined_documents = []
        seen_documents = set()

        # Search ChromaDB for each query
        for search_query in all_queries:

            results = self.retriever.similarity_search(
                query=search_query,
                k=k
            )

            for document in results:

                # Avoid returning duplicate chunks
                document_key = (
                    document.metadata.get("document_id"),
                    document.metadata.get("start_index"),
                    document.page_content
                )

                if document_key not in seen_documents:

                    seen_documents.add(document_key)
                    combined_documents.append(document)

        return {
            "queries": all_queries,
            "documents": combined_documents
        }
