import pickle
import tempfile
import logging

from scystream.sdk.core import entrypoint
from scystream.sdk.env.settings import (
    EnvSettings,
    InputSettings,
    OutputSettings,
    FileSettings
)
from scystream.sdk.file_handling.s3_manager import S3Operations

from preprocessing.core import Preprocessor
from preprocessing.loader import TxtLoader, BibLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DTMFileOutput(FileSettings, OutputSettings):
    __identifier__ = "dtm_output"

    FILE_EXT: str = "pkl"


class VocabFileOutput(FileSettings, OutputSettings):
    __identifier__ = "vocab_output"

    FILE_EXT: str = "pkl"


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
    dtm_output: DTMFileOutput
    vocab_output: VocabFileOutput


class PreprocessBIB(EnvSettings):
    LANGUAGE: str = "en"
    FILTER_STOPWORDS: bool = True
    UNIGRAM_NORMALIZER: str = "lemma"
    USE_NGRAMS: bool = True
    NGRAM_MIN: int = 2
    NGRAM_MAX: int = 3

    BIB_DOWNLOAD_PATH: str = "/tmp/input.bib"

    bib_input: BIBFileInput
    dtm_output: DTMFileOutput
    vocab_output: VocabFileOutput


def _preprocess_and_store(texts, settings):
    """Shared preprocessing logic for TXT and BIB."""
    logger.info(f"Starting preprocessing with {len(texts)} documents")

    pre = Preprocessor(
        language=settings.LANGUAGE,
        filter_stopwords=settings.FILTER_STOPWORDS,
        unigram_normalizer=settings.UNIGRAM_NORMALIZER,
        use_ngrams=settings.USE_NGRAMS,
        ngram_min=settings.NGRAM_MIN,
        ngram_max=settings.NGRAM_MAX,
    )

    pre.texts = texts
    pre.analyze_texts()

    pre.generate_bag_of_words()

    dtm, vocab = pre.generate_document_term_matrix()

    with tempfile.NamedTemporaryFile(suffix="_dtm.pkl") as tmp_dtm, \
            tempfile.NamedTemporaryFile(suffix="_vocab.pkl") as tmp_vocab:

        pickle.dump(dtm, tmp_dtm)
        tmp_dtm.flush()

        pickle.dump(vocab, tmp_vocab)
        tmp_vocab.flush()

        logger.info("Uploading DTM to S3...")
        S3Operations.upload(settings.dtm_output, tmp_dtm.name)

        logger.info("Uploading vocabulary to S3...")
        S3Operations.upload(settings.vocab_output, tmp_vocab.name)

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
