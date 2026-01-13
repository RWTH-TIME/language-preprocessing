import os
import tempfile

from preprocessing.loader import TxtLoader, BibLoader
from preprocessing.models import DocumentRecord


def test_txt_loader_reads_and_normalizes():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("Hello {World}\nSecond line")
        fname = f.name

    result = TxtLoader.load(fname)
    os.unlink(fname)

    # Expect list of DocumentRecord
    assert len(result) == 2

    assert isinstance(result[0], DocumentRecord)
    assert result[0].doc_id == "1"
    assert result[0].text == "Hello World"

    assert isinstance(result[1], DocumentRecord)
    assert result[1].doc_id == "2"
    assert result[1].text == "Second line"


def test_bib_loader_extracts_attribute():
    bib_content = r"""
    @article{a,
      abstract = {This is {Bib} \textbf{text}.},
      title = {Ignore me}
    }
    """

    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write(bib_content)
        fname = f.name

    result = BibLoader.load(fname, "abstract")
    os.unlink(fname)

    assert len(result) == 1

    record = result[0]
    assert isinstance(record, DocumentRecord)

    # ID taken from bib entry key: "@article{a,..."
    assert record.doc_id == "a"

    # Normalized abstract text
    assert record.text == "This is Bib text."
