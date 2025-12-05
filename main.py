import pickle
import tempfile

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

    bib_input: BIBFileInput
    dtm_output: DTMFileOutput
    vocab_output: VocabFileOutput


def _preprocess_and_store(texts, settings):
    """Shared preprocessing logic for TXT and BIB."""
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

        S3Operations.upload(settings.dtm_output, tmp_dtm.name)
        S3Operations.upload(settings.vocab_output, tmp_vocab.name)


@entrypoint(PreprocessTXT)
def preprocess_txt_file(settings):
    S3Operations.download(settings.txt_input, "input.txt")
    texts = TxtLoader.load("./input.txt")
    _preprocess_and_store(texts, settings)


@entrypoint(PreprocessBIB)
def preprocess_bib_file(settings):
    S3Operations.download(settings.bib_input, "input.bib")
    texts = BibLoader.load(
        "./input.bib",
        attribute=settings.bib_input.SELECTED_ATTRIBUTE,
    )
    _preprocess_and_store(texts, settings)


"""
if __name__ == "__main__":
    test = PreprocessBIB(
        bib_input=BIBFileInput(
            S3_HOST="http://localhost",
            S3_PORT="9000",
            S3_ACCESS_KEY="minioadmin",
            S3_SECRET_KEY="minioadmin",
            BUCKET_NAME="input-bucket",
            FILE_PATH="input_file_path",
            FILE_NAME="wos_export",
            SELECTED_ATTRIBUTE="abstract"
        ),
        dtm_output=DTMFileOutput(
            S3_HOST="http://localhost",
            S3_PORT="9000",
            S3_ACCESS_KEY="minioadmin",
            S3_SECRET_KEY="minioadmin",
            BUCKET_NAME="output-bucket",
            FILE_PATH="output_file_path",
            FILE_NAME="dtm_file_bib"
        ),
        vocab_output=VocabFileOutput(
            S3_HOST="http://localhost",
            S3_PORT="9000",
            S3_ACCESS_KEY="minioadmin",
            S3_SECRET_KEY="minioadmin",
            BUCKET_NAME="output-bucket",
            FILE_PATH="output_file_path",
            FILE_NAME="vocab_file_bib"
        )
    )

    preprocess_bib_file(test)
"""
