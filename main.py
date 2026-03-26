import hashlib
import logging

import pandas as pd
from preprocessing.core import Preprocessor
from preprocessing.loader import BibLoader, TxtLoader
from preprocessing.models import DocumentRecord, PreprocessedDocument
from scystream.sdk.core import entrypoint
from scystream.sdk.env.settings import (
    EnvSettings,
    FileSettings,
    InputSettings,
    OutputSettings,
    PostgresSettings,
)
from scystream.sdk.file_handling.s3_manager import S3Operations
from sqlalchemy import create_engine
from sqlalchemy.sql import quoted_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _normalize_table_name(table_name: str) -> str:
    max_length = 63
    if len(table_name) <= max_length:
        return table_name
    digest = hashlib.sha1(table_name.encode("utf-8")).hexdigest()[:10]
    prefix_length = max_length - len(digest) - 1
    return f"{table_name[:prefix_length]}_{digest}"


def _resolve_db_table(settings: PostgresSettings) -> str:
    normalized_name = _normalize_table_name(settings.DB_TABLE)
    settings.DB_TABLE = normalized_name
    return normalized_name


class NormalizedDocsOutput(PostgresSettings, OutputSettings):
    __identifier__ = "normalized_docs"


class TXTFileInput(FileSettings, InputSettings):
    __identifier__ = "txt_file"
    FILE_EXT: str = "txt"


class BIBFileInput(FileSettings, InputSettings):
    __identifier__ = "bib_file"
    FILE_EXT: str = "bib"

    SELECTED_ATTRIBUTE: str = "Abstract"


class PreprocessTXT(EnvSettings):
    LANGUAGE: str = "en"
    FILTER_STOPWORDS: bool = True
    UNIGRAM_NORMALIZER: str = "lemma"
    USE_NGRAMS: bool = True
    NGRAM_MIN: int = 2
    NGRAM_MAX: int = 3

    TXT_DOWNLOAD_PATH: str = "/tmp/input.txt"

    txt_input: TXTFileInput
    normalized_docs_output: NormalizedDocsOutput


class PreprocessBIB(EnvSettings):
    LANGUAGE: str = "en"
    FILTER_STOPWORDS: bool = True
    UNIGRAM_NORMALIZER: str = "lemma"
    USE_NGRAMS: bool = True
    NGRAM_MIN: int = 2
    NGRAM_MAX: int = 3

    BIB_DOWNLOAD_PATH: str = "/tmp/input.bib"

    bib_input: BIBFileInput
    normalized_docs_output: NormalizedDocsOutput


def _write_preprocessed_docs_to_postgres(
    preprocessed_ouput: list[PreprocessedDocument],
    settings: PostgresSettings,
):
    resolved_table_name = _resolve_db_table(settings)
    df = pd.DataFrame(
        [
            {
                "doc_id": d.doc_id,
                "tokens": d.tokens,
            }
            for d in preprocessed_ouput
        ],
    )

    logger.info(
        "Writing %s processed documents to DB table '%s'…",
        len(df),
        resolved_table_name,
    )
    engine = create_engine(
        f"postgresql+psycopg2://{settings.PG_USER}:{settings.PG_PASS}"
        f"@{settings.PG_HOST}:{int(settings.PG_PORT)}/",
    )

    table_name = quoted_name(resolved_table_name, quote=True)
    df.to_sql(table_name, engine, if_exists="replace", index=False)

    logger.info(
        "Successfully stored normalized documents into '%s'.",
        resolved_table_name,
    )


def _preprocess_and_store(documents: list[DocumentRecord], settings):
    """Shared preprocessing logic for TXT and BIB."""
    logger.info(f"Starting preprocessing with {len(documents)} documents")

    pre = Preprocessor(
        language=settings.LANGUAGE,
        filter_stopwords=settings.FILTER_STOPWORDS,
        unigram_normalizer=settings.UNIGRAM_NORMALIZER,
        use_ngrams=settings.USE_NGRAMS,
        ngram_min=settings.NGRAM_MIN,
        ngram_max=settings.NGRAM_MAX,
    )

    pre.documents = documents
    result = pre.generate_normalized_output()

    _write_preprocessed_docs_to_postgres(
        result,
        settings.normalized_docs_output,
    )

    logger.info("Preprocessing completed successfully.")


@entrypoint(PreprocessTXT)
def preprocess_txt_file(settings):
    logger.info("Downloading TXT input from S3...")
    S3Operations.download(settings.txt_input, settings.TXT_DOWNLOAD_PATH)

    texts = TxtLoader.load(settings.TXT_DOWNLOAD_PATH)

    _preprocess_and_store(texts, settings)


@entrypoint(PreprocessBIB)
def preprocess_bib_file(settings):
    logger.info("Downloading BIB input from S3...")
    S3Operations.download(settings.bib_input, settings.BIB_DOWNLOAD_PATH)

    texts = BibLoader.load(
        settings.BIB_DOWNLOAD_PATH,
        attribute=settings.bib_input.SELECTED_ATTRIBUTE,
    )
    _preprocess_and_store(texts, settings)
