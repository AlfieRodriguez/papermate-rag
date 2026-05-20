from pathlib import Path

import fitz
import pytest

from papermate.ingestion.pdf_loader import PDFLoader
from papermate.schemas import PageText


def create_pdf(path: Path, page_texts: list[str]) -> None:
    document = fitz.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_load_returns_page_text_objects(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    create_pdf(pdf_path, ["First page text.", "Second page text."])

    pages = PDFLoader().load(str(pdf_path), doc_id="paper-1")

    assert len(pages) == 2
    assert all(isinstance(page, PageText) for page in pages)
    assert [page.page_number for page in pages] == [1, 2]
    assert [page.doc_id for page in pages] == ["paper-1", "paper-1"]
    assert "First page text." in pages[0].text
    assert "Second page text." in pages[1].text


def test_load_includes_page_metadata(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    create_pdf(pdf_path, ["Metadata page."])

    page = PDFLoader().load(str(pdf_path), doc_id="paper-1")[0]

    assert page.metadata["file_path"] == str(pdf_path)
    assert page.metadata["file_name"] == "paper.pdf"
    assert page.metadata["page_count"] == 1


def test_missing_file_raises_file_not_found(tmp_path) -> None:
    missing_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        PDFLoader().load(str(missing_path), doc_id="paper-1")


def test_empty_file_path_raises_value_error() -> None:
    with pytest.raises(ValueError, match="file_path must not be empty"):
        PDFLoader().load("  ", doc_id="paper-1")


def test_empty_doc_id_raises_value_error(tmp_path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    create_pdf(pdf_path, ["Some text."])

    with pytest.raises(ValueError, match="doc_id must not be empty"):
        PDFLoader().load(str(pdf_path), doc_id="  ")


def test_directory_path_raises_value_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="PDF path is not a file"):
        PDFLoader().load(str(tmp_path), doc_id="paper-1")


def test_invalid_pdf_raises_value_error(tmp_path) -> None:
    invalid_pdf = tmp_path / "invalid.pdf"
    invalid_pdf.write_text("not a pdf", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not open PDF file"):
        PDFLoader().load(str(invalid_pdf), doc_id="paper-1")


def test_empty_pdf_page_is_returned_with_empty_text(tmp_path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    create_pdf(pdf_path, [""])

    pages = PDFLoader().load(str(pdf_path), doc_id="paper-1")

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].text == ""
