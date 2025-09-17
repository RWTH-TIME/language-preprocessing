import re
import bibtexparser


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Remove curly braces
    text = re.sub(r"[{}]", "", text)

    # Remove LaTeX commands
    text = re.sub(r"\\[a-zA-Z]+\s*(\{[^}]*\})?", "", text)

    # Remove LaTeX escaped quotes/accents
    text = re.sub(r"\\""[a-zA-Z]", lambda m: m.group(0)[-1], text)

    text = re.sub(r"\\'", "", text)
    text = text.replace("'", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


class TxtLoader:
    @staticmethod
    def load(file_path: str) -> list[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return [normalize_text(line) for line in lines]


class BibLoader:
    @staticmethod
    def load(file_path: str, attribute: str) -> list[str]:
        with open(file_path, "r", encoding="utf-8") as f:
            bib_database = bibtexparser.load(f)

        results = []
        for entry in bib_database.entries:
            value = entry.get(attribute.lower(), "")
            results.append(normalize_text(value))

        return results
