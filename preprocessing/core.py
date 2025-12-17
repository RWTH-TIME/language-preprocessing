import logging
import spacy

from typing import Literal, List
from nltk.stem.porter import PorterStemmer
from preprocessing.models import PreprocessedDocument, DocumentRecord

LANG_TO_SPACY_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm"
}
logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(
        self,
        language: Literal["de", "en"] = "en",
        filter_stopwords: bool = True,
        unigram_normalizer: Literal["lemma", "porter"] = "lemma",
        use_ngrams: bool = True,
        ngram_min: int = 2,
        ngram_max: int = 3,
    ):
        logger.info(
            "Init Preprocessor (lang=%s, filter_stopwords=%s, ngrams=%s)",
            language,
            filter_stopwords,
            use_ngrams,
        )
        self.language = language
        self.filter_stopwords = filter_stopwords
        self.unigram_normalizer = unigram_normalizer
        self.use_ngrams = use_ngrams
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

        self.nlp_model = LANG_TO_SPACY_MODELS.get(language, "en_core_web_sm")
        try:
            self.nlp = spacy.load(self.nlp_model, disable=["ner"])
        except OSError:
            spacy.cli.download(self.nlp_model)
            self.nlp = spacy.load(self.nlp_model, disable=["ner"])

        self.documents: List[DocumentRecord] = []

    def filter_tokens(
        self,
        tokens: list[spacy.tokens.Token],
        filter_stopwords: bool = False
    ) -> list[spacy.tokens.Token]:
        return [
            t for t in tokens
            if t.is_alpha
            and (not filter_stopwords or not t.is_stop)
            and len(t.text) > 2
        ]

    def generate_normalized_output(self) -> List[PreprocessedDocument]:
        logger.info("Generating normalized output...")
        porter = PorterStemmer()

        processed_docs: List[PreprocessedDocument] = []

        for record in self.documents:
            doc = self.nlp(record.text)
            doc_terms = []

            # Process each sentence
            for sent in doc.sents:
                filtered = self.filter_tokens(
                    list(sent), self.filter_stopwords
                )
                normalized = [
                    self.normalize_token(t, porter) for t in filtered
                ]
                doc_terms.extend(normalized)

                # Generate n-grams
                if self.use_ngrams and self.ngram_min > 1:
                    for n in range(self.ngram_min, self.ngram_max + 1):
                        for i in range(len(normalized) - n + 1):
                            ngram = " ".join(normalized[i:i+n])
                            doc_terms.append(ngram)

            processed_docs.append(PreprocessedDocument(
                doc_id=record.doc_id,
                tokens=doc_terms
            ))

        return processed_docs

    def normalize_token(
        self,
        token: spacy.tokens.Token,
        porter: PorterStemmer
    ):
        """Apply lemma or stem normalization."""
        word = token.text.lower() if not token.text.isupper() else token.text
        if word.isupper():
            return word
        if self.unigram_normalizer == "porter":
            return porter.stem(word)
        elif self.unigram_normalizer == "lemma":
            return token.lemma_.lower()
        return word
