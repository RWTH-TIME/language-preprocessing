from preprocessing.loader import normalize_text


def test_normalize_removes_braces():
    assert normalize_text("{abc}") == "abc"


def test_normalize_removes_latex_commands():
    assert normalize_text(r"\textbf{Hello}") == "Hello"


def test_normalize_removes_accents():
    assert normalize_text(r"\'a") == "a"


def test_normalize_collapses_whitespace():
    assert normalize_text("a    b   c") == "a b c"
