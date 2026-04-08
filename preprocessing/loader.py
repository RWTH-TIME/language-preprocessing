import logging
import re
import bibtexparser
from typing import List
from pathlib import Path

from preprocessing.models import DocumentRecord, PreprocessedDocument

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r'\\"([a-zA-Z])', r"\1", text)
    text = re.sub(r"\\'", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


class TxtLoader:
    @staticmethod
    def load(file_path: str) -> list[DocumentRecord]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        return [
            DocumentRecord(doc_id=str(i), text=normalize_text(line))
            for i, line in enumerate(lines, start=1)
        ]

    @staticmethod
    def overwrite_with_results(
        preprocessed_docs: List[PreprocessedDocument],
        export_path: Path,
    ) -> None:
        logger.info("Writing preprocessed TXT file...")

        output_path = Path.cwd() / export_path.name

        # Ensure correct order (IDs are numeric strings)
        sorted_docs = sorted(preprocessed_docs, key=lambda d: int(d.doc_id))

        with open(output_path, "w", encoding="utf-8") as f:
            for doc in sorted_docs:
                line = " ".join(doc.tokens)
                f.write(line + "\n")

        logger.info(f"TXT file successfully written to: {output_path}")


class BibLoader:
    def __init__(self, file_path: str, attribute: str):
        logger.info(f"Loading BIB file (attribute={attribute})...")

        with open(file_path, "r", encoding="utf-8") as f:
            self.bib_db = bibtexparser.load(f)

        self.file_path = file_path
        self.attribute = attribute.lower()

        self.document_records = self._build_document_records()

    @staticmethod
    def _extract_bib_id(entry: dict) -> str:
        return (
            entry.get("id")
            or entry.get("ID")
            or entry.get("citekey")
            or entry.get("entrykey")
            or entry.get("Unique-ID")
            or "UNKNOWN_ID"
        )

    def _build_document_records(self) -> List[DocumentRecord]:
        records = []

        for entry in self.bib_db.entries:
            bib_id = self._extract_bib_id(entry)
            raw_value = entry.get(self.attribute, "")
            normalized = normalize_text(raw_value)

            records.append(DocumentRecord(doc_id=bib_id, text=normalized))

        return records

    def overwrite_with_results(
        self, preprocessed_docs: List[PreprocessedDocument], export_path: Path
    ) -> None:
        logger.info("Overwriting input documents with preprocessed text...")

        output_path = Path.cwd() / export_path.name

        preprocessed_dict = {doc.doc_id: doc for doc in preprocessed_docs}

        for entry in self.bib_db.entries:
            bib_id = self._extract_bib_id(entry)
            preprocessed = preprocessed_dict.get(bib_id)

            if not preprocessed:
                continue

            entry[self.attribute] = " ".join(preprocessed.tokens)

        with open(output_path, "w", encoding="utf-8") as f:
            bibtexparser.dump(self.bib_db, f)

        logger.info(f"BIB file successfully written to: {output_path}")
