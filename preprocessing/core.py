import spacy
import numpy as np

from typing import Literal
from nltk.stem.porter import PorterStemmer
from collections import Counter

LANG_TO_SPACY_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm"
}


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
        self.language = language
        self.filter_stopwords = filter_stopwords
        self.unigram_normalizer = unigram_normalizer
        self.use_ngrams = use_ngrams
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

        self.nlp_model = LANG_TO_SPACY_MODELS.get(language, "en_core_web_sm")

        self.ngram_frequency = Counter()
        self.ngram_document_frequency = Counter()
        self.token_frequency = Counter()
        self.token_document_frequency = Counter()

        self.texts: list[str] = []

        self.bag_of_words = []

        try:
            self.nlp = spacy.load(self.nlp_model, disable=["ner"])
        except OSError:
            spacy.cli.download(self.nlp_model)
            self.nlp = spacy.load(self.nlp_model, disable=["ner"])

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

    def analyze_texts(self):
        porter = PorterStemmer()
        for text in self.texts:
            doc = self.nlp(text)

            token_list = []
            ngram_list = []

            for sentence in doc.sents:
                filtered_tokens = self.filter_tokens(
                        list(sentence),
                        self.filter_stopwords
                )
                normalized_tokens = [
                    self.normalize_token(t, porter) for t in filtered_tokens
                ]
                token_list.extend(normalized_tokens)

                if (
                    self.use_ngrams and
                    self.ngram_min > 1 and
                    self.ngram_max > 1
                ):
                    for n in range(self.ngram_min, self.ngram_max + 1):
                        for i in range(len(normalized_tokens) - n + 1):
                            ngram = " ".join(normalized_tokens[i:i+n])
                            ngram_list.append(ngram)

            # update unigram counters
            self.token_frequency.update(token_list)
            self.token_document_frequency.update(set(token_list))

            # update n-gram counters if any
            if ngram_list:
                self.ngram_frequency.update(ngram_list)
                self.ngram_document_frequency.update(set(ngram_list))

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

    def generate_bag_of_words(self):
        porter = PorterStemmer()
        self.bag_of_words = []

        for text in self.texts:
            doc = self.nlp(text)
            doc_terms = []

            for sent in doc.sents:
                tokens = self.filter_tokens(list(sent), self.filter_stopwords)

                # Handle unigrams
                for token in tokens:
                    normalized = self.normalize_token(token, porter)

                    token_dict = {
                        "term": normalized,
                        "type": "word",
                        "span": 1,
                        "freq": self.token_frequency.get(normalized, 0),
                        "docs": (
                            self.token_document_frequency.get(normalized, 0)
                        ),
                        "filters": (
                            ["stop"] if not self.filter_stopwords
                            and token.is_stop else []
                        )
                    }
                    doc_terms.append(token_dict)

                # Handle ngrams
                if self.use_ngrams and self.ngram_min > 1:
                    added_ngrams = set()  # avoid duplicates
                    for n in range(self.ngram_min, self.ngram_max + 1):
                        for i in range(len(tokens) - n + 1):
                            ngram_tokens = tokens[i:i+n]
                            ngram_str = " ".join(
                                [self.normalize_token(t, porter)
                                 for t in ngram_tokens]
                            )

                            if ngram_str in added_ngrams:
                                continue
                            added_ngrams.add(ngram_str)

                            ngram_dict = {
                                "term": ngram_str,
                                "type": "ngram",
                                "span": n,
                                "freq": self.ngram_frequency.get(ngram_str, 0),
                                "docs": (
                                    self.ngram_document_frequency.get(
                                        ngram_str, 0
                                    )
                                ),
                                "filters": []
                            }
                            doc_terms.append(ngram_dict)

            self.bag_of_words.append(doc_terms)

    def generate_document_term_matrix(self) -> (np.ndarray, dict):
        """
        Converts bag_of_words into document-term matrix

            dtm (np.ndarray): shape = (num_docs, num_terms)
            vocab (dict): mapping term -> column index
        """

        all_terms = set()
        for doc in self.bag_of_words:
            for t in doc:
                all_terms.add(t["term"])

        vocab = {term: idx for idx, term in enumerate(sorted(all_terms))}

        num_docs = len(self.bag_of_words)
        num_terms = len(vocab)
        dtm = np.zeros((num_docs, num_terms), dtype=int)

        for doc_idx, doc in enumerate(self.bag_of_words):
            for token in doc:
                term_idx = vocab[token["term"]]
                dtm[doc_idx, term_idx] += 1

        return dtm, vocab
