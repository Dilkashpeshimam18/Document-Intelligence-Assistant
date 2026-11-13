
from ingestion.document_loader import DocumentIngestion
from ingestion.document_cleaner import DocumentCleaner
from ingestion.metadata_extractor import MetadataExtractor

from chunking.text_splitter import RecursiveChunker

from embedding.embedding_model import EmbeddingModel

from vectorstore.chroma_store import ChromaVectorStore

from retrieval.retriever import DocumentRetriever
from retrieval.bm25_retriever import BM25DocumentRetriever
from retrieval.hybrid_search import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

def main():

    # ==========================================
    # 1. DOCUMENT INGESTION
    # ==========================================

    ingestion = DocumentIngestion(
        data_path="data/raw"
    )

    documents = ingestion.load_all_documents()

    print("\n========== DOCUMENT INGESTION ==========")
    print("Loaded documents:", len(documents))

    if not documents:
        print("No documents found. Add PDF, TXT or DOCX files to data/raw.")
        return

    # ==========================================
    # 2. DOCUMENT CLEANING
    # ==========================================

    cleaner = DocumentCleaner()

    cleaned_documents = cleaner.clean_documents(
        documents
    )

    print("\n========== DOCUMENT CLEANING ==========")
    print("Cleaned documents:", len(cleaned_documents))

    # ==========================================
    # 3. METADATA EXTRACTION
    # ==========================================

    extractor = MetadataExtractor()

    processed_documents = extractor.extract_metadata(
        cleaned_documents
    )

    print("\n========== METADATA EXTRACTION ==========")
    print("Processed documents:", len(processed_documents))

    # ==========================================
    # 4. DOCUMENT CHUNKING
    # ==========================================

    chunker = RecursiveChunker(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = chunker.split_documents(
        processed_documents
    )

    print("\n========== CHUNKING ==========")
    print("Generated chunks:", len(chunks))

    if not chunks:
        print("No chunks generated.")
        return

    # ==========================================
    # 5. EMBEDDING MODEL
    # ==========================================

    embedding_service = EmbeddingModel()

    print("\n========== EMBEDDING MODEL ==========")
    print("Embedding model initialized.")

    # ==========================================
    # 6. CHROMADB INITIALIZATION
    # ==========================================

    vector_db = ChromaVectorStore(
        embedding_model=embedding_service.model,
        persist_directory="./data/chroma_db",
        collection_name="document_intelligence"
    )

    # ==========================================
    # 7. VECTOR INDEXING
    # ==========================================

    ids = vector_db.add_documents(chunks)

    print("\n========== VECTOR DATABASE ==========")
    print("Indexed chunks:", len(ids))

    # ==========================================
    # 8. BM25 RETRIEVAL
    # ==========================================

    bm25_retriever = BM25DocumentRetriever(
        documents=chunks,
        k=4
    )

    # ==========================================
    # 9. HYBRID RETRIEVAL
    # ==========================================

    document_retriever = DocumentRetriever(
        vector_store=vector_db
    )

    hybrid_retriever = HybridRetriever(
        vector_retriever=document_retriever,
        bm25_retriever=bm25_retriever,
        rrf_constant=60
    )

    # ==========================================
    # 10. TEST RETRIEVAL
    # ==========================================

    

    # ==========================================
    # 10. HYBRID RETRIEVAL
    # ==========================================

    query = "How does supervised learning train a model?"

    hybrid_results = hybrid_retriever.retrieve(
        query=query,
        k=10
    )

    print("\n========== HYBRID SEARCH RESULTS ==========")

    for index, result in enumerate(hybrid_results):

        print(f"\nResult {index + 1}")
        print("RRF Score:", result["rrf_score"])
        print("Content:", result["document"].page_content[:250])

    # ==========================================
    # 11. CROSS-ENCODER RERANKING
    # ==========================================

    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu"
    )

    reranked_results = reranker.rerank(
        query=query,
        retrieved_results=hybrid_results,
        top_n=4
    )

    print("\n========== RERANKED RESULTS ==========")

    for index, result in enumerate(reranked_results):

        document = result["document"]

        print(f"\nResult {index + 1}")
        print("RRF Score:", result["rrf_score"])
        print("Reranker Score:", result["rerank_score"])
        print("Content:", document.page_content[:300])
        print("Metadata:", document.metadata)


if __name__ == "__main__":
    main()
