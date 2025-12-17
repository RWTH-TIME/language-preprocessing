def test_preprocessor_tokenization(preprocessor, simple_texts):
    preprocessor.texts = simple_texts
    preprocessor.analyze_texts()

    assert len(preprocessor.token_frequency) > 0


def test_preprocessor_bag_of_words(preprocessor, simple_texts):
    preprocessor.texts = simple_texts
    preprocessor.analyze_texts()
    preprocessor.generate_bag_of_words()

    assert len(preprocessor.bag_of_words) == 2
    assert all(len(doc) > 0 for doc in preprocessor.bag_of_words)


def test_generate_document_term_matrix(preprocessor, simple_texts):
    preprocessor.texts = simple_texts
    preprocessor.analyze_texts()
    preprocessor.generate_bag_of_words()

    dtm, vocab = preprocessor.generate_document_term_matrix()

    assert dtm.shape[0] == 2
    assert dtm.shape[1] == len(vocab)
    assert dtm.sum() > 0
