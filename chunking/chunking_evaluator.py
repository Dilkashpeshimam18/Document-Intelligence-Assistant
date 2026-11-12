
from statistics import mean


class ChunkingEvaluator:

    @staticmethod
    def evaluate(chunks, strategy_name):

        if not chunks:
            print(f"{strategy_name}: No chunks generated.")
            return {}

        chunk_lengths = [
            len(chunk.page_content)
            for chunk in chunks
        ]

        results = {
            "strategy": strategy_name,
            "total_chunks": len(chunks),
            "average_length": round(mean(chunk_lengths), 2),
            "minimum_length": min(chunk_lengths),
            "maximum_length": max(chunk_lengths),
        }

        return results

    @staticmethod
    def compare(results):

        print("\n--- CHUNKING COMPARISON ---")

        for result in results:
            print(f"\nStrategy: {result['strategy']}")
            print(f"Total chunks: {result['total_chunks']}")
            print(f"Average length: {result['average_length']}")
            print(f"Minimum length: {result['minimum_length']}")
            print(f"Maximum length: {result['maximum_length']}")
