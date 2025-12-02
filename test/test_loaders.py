import os
import tempfile

from preprocessing.loader import TxtLoader, BibLoader


def test_txt_loader_reads_and_normalizes():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("Hello {World}\nSecond line")
        fname = f.name

    result = TxtLoader.load(fname)
    os.unlink(fname)

    assert result == ["Hello World", "Second line"]


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

    assert result == ["This is Bib text."]
