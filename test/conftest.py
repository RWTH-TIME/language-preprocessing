import pytest

from preprocessing.core import Preprocessor


@pytest.fixture
def simple_texts():
    return ["This is a test sentence.", "Another test sentence."]


@pytest.fixture
def preprocessor():
    return Preprocessor(
        language="en",
        filter_stopwords=True,
        unigram_normalizer="porter",
        use_ngrams=True,
        ngram_min=2,
        ngram_max=3,
    )
