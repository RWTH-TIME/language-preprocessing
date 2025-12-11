import logging
import re
import bibtexparser

from preprocessing.models import DocumentRecord

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)

    text = re.sub(r"\\[a-zA-Z]+", "", text)

    text = re.sub(r"[{}]", "", text)

    text = re.sub(r'\\"([a-zA-Z])', r'\1', text)

    text = re.sub(r"\\'", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


class TxtLoader:
    @staticmethod
    def load(file_path: str) -> list[DocumentRecord]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        return [
            DocumentRecord(
                doc_id=str(i),
                text=normalize_text(line)
            )
            for i, line in enumerate(lines, start=1)
        ]


class BibLoader:
    @staticmethod
    def load(file_path: str, attribute: str) -> list[DocumentRecord]:
        logger.info(f"Loading BIB file (attribute={attribute})...")

        with open(file_path, "r", encoding="utf-8") as f:
            bib_database = bibtexparser.load(f)

        results = []
        attribute_lower = attribute.lower()

        for entry in bib_database.entries:
            bib_id = (
                entry.get("id")
                or entry.get("ID")
                or entry.get("citekey")
                or entry.get("entrykey")
                or entry.get("Unique-ID")
                or "UNKNOWN_ID"
            )

            raw_value = entry.get(attribute_lower, "")
            normalized = normalize_text(raw_value)

            results.append(DocumentRecord(
                doc_id=bib_id,
                text=normalized
            ))

        return results
