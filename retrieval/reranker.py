
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    ):

        self.model = CrossEncoder(
            model_name,
            device=device
        )

    def rerank(self, query, retrieved_results, top_n=4):

        if not retrieved_results:
            return []

        # Prepare query-document pairs
        pairs = [
            (
                query,
                result["document"].page_content
            )
            for result in retrieved_results
        ]

        # Predict relevance scores
        scores = self.model.predict(pairs)

        # Attach scores to retrieved documents
        reranked_results = []

        for result, score in zip(retrieved_results, scores):

            reranked_results.append({
                "document": result["document"],
                "rrf_score": result["rrf_score"],
                "rerank_score": float(score)
            })

        # Sort by cross-encoder relevance score
        reranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_n]
