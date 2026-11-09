
import re
import unicodedata

from langchain_core.documents import Document


class DocumentCleaner:

    def clean_text(self, text):

        # 1. Normalize Unicode characters
        text = unicodedata.normalize("NFC", text)

        # 2. Remove null characters
        text = text.replace("\x00", "")

        # 3. Remove unwanted control characters,
        # while preserving newlines and tabs
        text = re.sub(
            r"[^\x09\x0A\x0D\x20-\uFFFF]",
            "",
            text
        )

        # 4. Replace multiple spaces and tabs
        text = re.sub(r"[ \t]+", " ", text)

        # 5. Remove excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 6. Remove leading and trailing whitespace
        text = text.strip()

        return text

    def clean_documents(self, documents):

        cleaned_documents = []

        for doc in documents:

            cleaned_text = self.clean_text(
                doc.page_content
            )

            if not cleaned_text:
                continue

            cleaned_doc = Document(
                page_content=cleaned_text,
                metadata=doc.metadata.copy()
            )

            cleaned_documents.append(cleaned_doc)

        return cleaned_documents
