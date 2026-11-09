
import hashlib
import uuid

from datetime import datetime, timezone
from pathlib import Path


class MetadataExtractor:

    def extract_metadata(self, documents):

        processed_documents = []

        for doc in documents:

            # Get document source
            source = doc.metadata.get(
                "source", "unknown"
            )

            file_path = Path(source)

            # Extract file extension
            file_type = file_path.suffix.lower().replace(
                ".", ""
            )

            # Generate content hash
            content_hash = hashlib.sha256(
                doc.page_content.encode("utf-8")
            ).hexdigest()

            # Generate stable document ID
            page = doc.metadata.get("page", 0)

            document_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source}:{page}"
                )
            )

            # Additional metadata
            metadata = doc.metadata.copy()

            metadata.update({
                "document_id": document_id,
                "source": source,
                "file_name": file_path.name,
                "file_type": file_type,
                "content_hash": content_hash,
                "ingested_at": datetime.now(
                    timezone.utc
                ).isoformat()
            })

            # Update document metadata
            doc.metadata = metadata

            processed_documents.append(doc)

        return processed_documents
