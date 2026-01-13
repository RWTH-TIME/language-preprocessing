from typing import List
from dataclasses import dataclass


@dataclass
class DocumentRecord:
    doc_id: str        # "0", "1", ... for TXT OR bib_id for BIB
    text: str          # normalized text


@dataclass
class PreprocessedDocument:
    doc_id: str
    tokens: List[str]
