from ingestion.document_loader import DataIngestion
from ingestion.document_cleaner import DocumentCleaner
from ingestion.metadata_extractor import MetadataExtractor

ingestion=DataIngestion(
    data_path='data/raw'
)

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

print("Processed documents:", len(processed_documents))

# Display extracted content
for doc in processed_documents:

    print("\n----------------------------")

    print("Document ID:", doc.metadata["document_id"])

    print("Source:", doc.metadata["source"])

    print("File Type:", doc.metadata["file_type"])

    print("Content Hash:", doc.metadata["content_hash"])

    print("Content:", doc.page_content[:300])

    print("Metadata:", doc.metadata)