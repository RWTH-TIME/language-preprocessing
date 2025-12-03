import logging
import re
import bibtexparser

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
    def load(file_path: str) -> list[str]:
        logger.info("Loading TXT file...")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [normalize_text(line) for line in lines]


class BibLoader:
    @staticmethod
    def load(file_path: str, attribute: str) -> list[str]:
        logger.info(f"Loading BIB file (attribute={attribute})...")
        with open(file_path, "r", encoding="utf-8") as f:
            bib_database = bibtexparser.load(f)

        results = []
        for entry in bib_database.entries:
            value = entry.get(attribute.lower(), "")
            results.append(normalize_text(value))

        return results
