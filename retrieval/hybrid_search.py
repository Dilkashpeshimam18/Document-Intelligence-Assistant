
class HybridRetriever:

    def __init__(
        self,
        vector_retriever,
        bm25_retriever,
        rrf_constant=60
    ):

        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_constant = rrf_constant

    @staticmethod
    def _document_key(document):

        # Identify the same chunk across both retrieval systems
        return (
            document.metadata.get("document_id"),
            document.metadata.get("start_index"),
            document.page_content
        )

    def retrieve(self, query, k=4):

        # 1. Retrieve using semantic vector search
        vector_results = self.vector_retriever.similarity_search(
            query=query,
            k=k
        )

        # 2. Retrieve using BM25 keyword search
        bm25_results = self.bm25_retriever.search(query)

        # 3. Combine rankings using RRF
        scores = {}
        documents = {}

        result_lists = [
            vector_results,
            bm25_results
        ]

        for result_list in result_lists:

            seen_in_list = set()

            for rank, document in enumerate(result_list, start=1):

                key = self._document_key(document)

                # Don't count a duplicate twice in one ranking
                if key in seen_in_list:
                    continue

                seen_in_list.add(key)
                documents[key] = document

                scores[key] = scores.get(key, 0) + (
                    1 / (self.rrf_constant + rank)
                )

        # 4. Sort by combined RRF score
        ranked_results = sorted(
            documents.items(),
            key=lambda item: scores[item[0]],
            reverse=True
        )

        # 5. Return the top results
        return [
            {
                "document": document,
                "rrf_score": scores[key]
            }
            for key, document in ranked_results[:k]
        ]
