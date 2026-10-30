import hashlib
import json

import pymupdf
import pytest

from src.syllabus.pdf_extractor import (
    PDFExtractionError,
    extract_syllabus,
    save_syllabus_json,
    validate_syllabus_data,
)


def create_pdf(path, page_texts: list[str]) -> None:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_extracts_course_outcomes_units_topics_and_page_numbers(tmp_path) -> None:
    pdf_path = tmp_path / "course.pdf"
    create_pdf(
        pdf_path,
        [
            "Course Name: Computer Networks\n"
            "Course Code: CS305\n"
            "UNIT I: Network Fundamentals\n"
            "- Network models\n"
            "- Protocol layering\n"
            "CO1: Explain the layered network architecture.",
            "UNIT II - Data Link Layer\n"
            "Error detection\n"
            "Flow control\n"
            "CO2: Analyze error control mechanisms.",
        ],
    )

    data = extract_syllabus(pdf_path)

    assert data["course_name"] == "Computer Networks"
    assert data["course_code"] == "CS305"
    assert data["units"] == [
        {
            "unit_number": 1,
            "unit_title": "Network Fundamentals",
            "topics": ["Network models", "Protocol layering"],
            "source_pages": [1],
        },
        {
            "unit_number": 2,
            "unit_title": "Data Link Layer",
            "topics": ["Error detection", "Flow control"],
            "source_pages": [2],
        },
    ]
    assert data["course_outcomes"] == [
        {
            "id": "CO1",
            "description": "Explain the layered network architecture.",
            "source_pages": [1],
        },
        {
            "id": "CO2",
            "description": "Analyze error control mechanisms.",
            "source_pages": [2],
        },
    ]
    assert data["review_status"] == "pending"
    validate_syllabus_data(data)


def test_roman_unit_number_and_multiline_outcome_are_supported(tmp_path) -> None:
    pdf_path = tmp_path / "roman.pdf"
    create_pdf(
        pdf_path,
        [
            "Unit IV: Transport Layer\n"
            "Congestion control\n"
            "Course Outcome 1: Describe transport protocols\n"
            "and explain their reliability mechanisms.",
        ],
    )

    data = extract_syllabus(pdf_path)

    assert data["units"][0]["unit_number"] == 4
    assert data["course_outcomes"][0]["id"] == "CO1"
    assert (
        data["course_outcomes"][0]["description"]
        == "Describe transport protocols and explain their reliability mechanisms."
    )


def test_missing_metadata_is_flagged_instead_of_invented(tmp_path) -> None:
    pdf_path = tmp_path / "generic.pdf"
    create_pdf(pdf_path, ["A general introduction to this document."])

    data = extract_syllabus(pdf_path)

    assert data["course_name"] is None
    assert data["course_code"] is None
    assert not data["units"]
    assert not data["course_outcomes"]
    assert any("name was not clearly identified" in warning for warning in data["warnings"])
    assert any("outcomes were found" in warning for warning in data["warnings"])


def test_empty_or_scanned_pdf_reports_no_selectable_text(tmp_path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    create_pdf(pdf_path, [""])

    with pytest.raises(PDFExtractionError, match="No selectable text"):
        extract_syllabus(pdf_path)


def test_password_protected_pdf_reports_that_it_cannot_be_read(tmp_path) -> None:
    pdf_path = tmp_path / "protected.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Course Code: CS305")
    document.save(
        pdf_path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    document.close()

    with pytest.raises(PDFExtractionError, match="password-protected"):
        extract_syllabus(pdf_path)


def test_invalid_and_missing_pdf_are_reported(tmp_path) -> None:
    with pytest.raises(PDFExtractionError, match="does not exist"):
        extract_syllabus(tmp_path / "missing.pdf")

    text_path = tmp_path / "notes.txt"
    text_path.write_text("not a PDF", encoding="utf-8")
    with pytest.raises(PDFExtractionError, match="must be a PDF"):
        extract_syllabus(text_path)

    broken_pdf = tmp_path / "broken.pdf"
    broken_pdf.write_bytes(b"not a PDF document")
    with pytest.raises(PDFExtractionError, match="Could not open"):
        extract_syllabus(broken_pdf)


def test_reviewed_json_is_saved_and_original_pdf_is_preserved(tmp_path) -> None:
    pdf_path = tmp_path / "Network Syllabus.pdf"
    create_pdf(
        pdf_path,
        ["Course Name: Networks\nCourse Code: CS101\nCO1: Explain networking."],
    )
    original_digest = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    data = extract_syllabus(pdf_path)
    data["review_status"] = "reviewed"
    output_directory = tmp_path / "parsed"

    saved_path = save_syllabus_json(data, output_directory)

    assert saved_path.parent == output_directory
    assert saved_path.is_file()
    assert saved_path.suffix == ".json"
    saved_data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert saved_data["review_status"] == "reviewed"
    assert saved_data["source_file"] == str(pdf_path.resolve())
    assert hashlib.sha256(pdf_path.read_bytes()).hexdigest() == original_digest


def test_validator_rejects_duplicate_outcome_ids(tmp_path) -> None:
    pdf_path = tmp_path / "duplicate.pdf"
    create_pdf(pdf_path, ["Course Outcome"])
    data = {
        "course_name": "Example",
        "course_code": "CS101",
        "units": [],
        "course_outcomes": [
            {"id": "CO1", "description": "First", "source_pages": [1]},
            {"id": "co1", "description": "Duplicate", "source_pages": [1]},
        ],
        "pages": [{"page": 1, "text": "Course Outcome"}],
        "source_file": str(pdf_path),
        "review_status": "reviewed",
        "warnings": [],
    }

    with pytest.raises(ValueError, match="Duplicate course outcome"):
        validate_syllabus_data(data)


def test_validator_rejects_source_page_outside_pdf(tmp_path) -> None:
    pdf_path = tmp_path / "source-page.pdf"
    create_pdf(pdf_path, ["Course Outcome"])
    data = {
        "course_name": "Example",
        "course_code": "CS101",
        "units": [
            {
                "unit_number": 1,
                "unit_title": "Introduction",
                "topics": ["Topic"],
                "source_pages": [2],
            }
        ],
        "course_outcomes": [],
        "pages": [{"page": 1, "text": "Course Outcome"}],
        "source_file": str(pdf_path),
        "review_status": "reviewed",
        "warnings": [],
    }

    with pytest.raises(ValueError, match="page not present in the PDF"):
        validate_syllabus_data(data)
