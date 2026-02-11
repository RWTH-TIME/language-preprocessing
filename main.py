import logging
import pandas as pd
from sqlalchemy import create_engine

from pathlib import Path
from typing import List
from scystream.sdk.core import entrypoint
from scystream.sdk.env.settings import (
    EnvSettings,
    InputSettings,
    OutputSettings,
    FileSettings,
    PostgresSettings
)
from scystream.sdk.file_handling.s3_manager import S3Operations

from preprocessing.core import Preprocessor
from preprocessing.loader import TxtLoader, BibLoader
from preprocessing.models import DocumentRecord, PreprocessedDocument

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NormalizedDocsOutput(PostgresSettings, OutputSettings):
    __identifier__ = "normalized_docs"


class NormalizedTXTOutput(FileSettings, OutputSettings):
    __identifier__ = "normalized_overwritten_file_output"
    FILE_EXT: str = "txt"


class TXTFileInput(FileSettings, InputSettings):
    __identifier__ = "txt_file"
    FILE_EXT: str = "txt"


class BIBFileInput(FileSettings, InputSettings):
    __identifier__ = "bib_file"
    FILE_EXT: str = "bib"

    SELECTED_ATTRIBUTE: str = "Abstract"


class NormalizedBIBOutput(FileSettings, OutputSettings):
    __identifier__ = "normalized_overwritten_file_output"
    FILE_EXT: str = "bib"


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
    normalized_overwritten_file_output: NormalizedTXTOutput


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
    normalized_overwritten_file_output: NormalizedBIBOutput


def _write_preprocessed_docs_to_postgres(
        preprocessed_ouput: List[PreprocessedDocument],
        settings: PostgresSettings
):
    df = pd.DataFrame([
        {
            "doc_id": d.doc_id,
            "tokens": d.tokens
        }
        for d in preprocessed_ouput
    ])

    logger.info(f"Writing {len(df)} processed documents to DB table '{
                settings.DB_TABLE}'…")
    engine = create_engine(
        f"postgresql+psycopg2://{settings.PG_USER}:{settings.PG_PASS}"
        f"@{settings.PG_HOST}:{int(settings.PG_PORT)}/"
    )

    df.to_sql(settings.DB_TABLE, engine, if_exists="replace", index=False)

    logger.info(f"Successfully stored normalized documents into '{
                settings.DB_TABLE}'.")


def _preprocess_and_store(
    documents: List[DocumentRecord],
    overwrite_callback,
    settings,
) -> List[PreprocessedDocument]:

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
        settings.normalized_docs_output
    )

    # Overwrite file using injected behavior
    export_path = Path(f"output.{
        settings.normalized_overwritten_file_output.FILE_EXT}")
    overwrite_callback(result, export_path)

    S3Operations.upload(
        settings.normalized_overwritten_file_output,
        export_path
    )

    logger.info("Preprocessing completed successfully.")
    return result


@entrypoint(PreprocessTXT)
def preprocess_txt_file(settings):
    logger.info("Downloading TXT file...")
    S3Operations.download(settings.txt_input, settings.TXT_DOWNLOAD_PATH)

    documents = TxtLoader.load(settings.TXT_DOWNLOAD_PATH)

    _preprocess_and_store(
        documents=documents,
        overwrite_callback=TxtLoader.overwrite_with_results,
        settings=settings
    )


@entrypoint(PreprocessBIB)
def preprocess_bib_file(settings):
    logger.info("Downloading BIB file...")
    S3Operations.download(settings.bib_input, settings.BIB_DOWNLOAD_PATH)

    loader = BibLoader(
        file_path=settings.BIB_DOWNLOAD_PATH,
        attribute=settings.bib_input.SELECTED_ATTRIBUTE
    )

    _preprocess_and_store(
        documents=loader.document_records,
        overwrite_callback=loader.overwrite_with_results,
        settings=settings
    )
