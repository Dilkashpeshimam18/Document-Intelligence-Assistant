
import logging

from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class DocumentIngestion:

    SUPPORTED_FORMATS = {
        ".pdf",
        ".txt",
        ".docx"
    }

    def __init__(
        self,
        data_path="data/raw",
        max_file_size_mb=50
    ):

        self.data_path = Path(data_path)

        self.max_file_size = max_file_size_mb * 1024 * 1024

        self.stats = {
            "total_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "skipped_files": 0,
            "total_documents": 0
        }

    # Validate files before loading
    def validate_file(self, file_path):

        if not file_path.is_file():
            logger.warning(f"Not a file: {file_path}")
            return False

        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            logger.warning(
                f"Unsupported format: {file_path.name}"
            )
            return False

        if file_path.stat().st_size == 0:
            logger.warning(
                f"Empty file: {file_path.name}"
            )
            return False

        if file_path.stat().st_size > self.max_file_size:
            logger.warning(
                f"File too large: {file_path.name}"
            )
            return False

        return True

    # Load PDF
    def load_pdf(self, file_path):

        loader = PyPDFLoader(str(file_path))

        return loader.load()

    # Load TXT
    def load_txt(self, file_path):

        loader = TextLoader(
            str(file_path),
            encoding="utf-8"
        )

        return loader.load()

    # Load DOCX
    def load_docx(self, file_path):

        loader = Docx2txtLoader(str(file_path))

        return loader.load()

    # Select loader based on extension
    def load_single_document(self, file_path):

        extension = file_path.suffix.lower()

        if extension == ".pdf":
            return self.load_pdf(file_path)

        elif extension == ".txt":
            return self.load_txt(file_path)

        elif extension == ".docx":
            return self.load_docx(file_path)

        return []

    # Load all documents
    def load_all_documents(self):

        documents = []

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_path}"
            )

        if not self.data_path.is_dir():
            raise NotADirectoryError(
                f"Expected a directory: {self.data_path}"
            )

        # Scan files recursively
        for file_path in self.data_path.rglob("*"):

            if not file_path.is_file():
                continue

            # Ignore hidden files
            if file_path.name.startswith("."):
                continue

            self.stats["total_files"] += 1

            try:

                # Validate file
                if not self.validate_file(file_path):

                    self.stats["skipped_files"] += 1
                    continue

                # Load document
                docs = self.load_single_document(file_path)

                # Check extracted content
                docs = [
                    doc for doc in docs
                    if doc.page_content.strip()
                ]

                if not docs:

                    logger.warning(
                        f"No text extracted: {file_path.name}"
                    )

                    self.stats["skipped_files"] += 1
                    continue

                documents.extend(docs)

                self.stats["successful_files"] += 1

                self.stats["total_documents"] += len(docs)

                logger.info(
                    f"Successfully loaded: {file_path.name}"
                )

            except Exception as e:

                self.stats["failed_files"] += 1

                logger.exception(
                    f"Failed to load {file_path.name}: {e}"
                )

                continue

        logger.info(
            f"Ingestion completed. Statistics: {self.stats}"
        )

        return documents
