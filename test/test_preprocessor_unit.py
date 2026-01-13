from preprocessing.core import Preprocessor
from preprocessing.models import DocumentRecord


def test_preprocessor_generate_normalized_output():
    # Prepare input documents as dataclasses
    docs = [
        DocumentRecord(doc_id="1", text="Dogs are running fast."),
        DocumentRecord(doc_id="2", text="Cats jump high.")
    ]

    pre = Preprocessor(
        language="en",
        filter_stopwords=True,
        unigram_normalizer="lemma",
        use_ngrams=True,
        ngram_min=2,
        ngram_max=2,
    )

    pre.documents = docs
    output = pre.generate_normalized_output()

    # Basic structure checks
    assert len(output) == 2
    assert output[0].doc_id == "1"
    assert output[1].doc_id == "2"

    # Tokens must not be empty
    assert len(output[0].tokens) > 0
    assert len(output[1].tokens) > 0

    # Check that lemmatization worked
    # "running" → "run"
    assert "run" in output[0].tokens

    # Stopwords filtered → "are" removed
    assert "are" not in output[0].tokens

    # Check n-gram generation (bigram because ngram_min=ngram_max=2)
    # Example bigram from doc1: "dog run" (if spacy lemmatizes)
    bigrams_doc1 = [tok for tok in output[0].tokens if " " in tok]
    assert len(bigrams_doc1) > 0  # at least one n-gram produced
