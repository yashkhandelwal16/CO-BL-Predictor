import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pymupdf


class PDFExtractionError(ValueError):
    """Raised when a syllabus PDF cannot be read as a text-based PDF."""


_UNIT_HEADING = re.compile(
    r"^\s*(?:unit|module)\s*(?:no\.?\s*)?[-:]?\s*"
    r"(?P<number>\d+|[ivxlcdm]+)\b\s*[:.)\-–—]?\s*(?P<title>.*)$",
    re.IGNORECASE,
)
_COURSE_OUTCOME = re.compile(
    r"^\s*(?:CO\s*[-:]?\s*(?P<co_number>\d+)|"
    r"course\s+(?:learning\s+)?outcomes?\s*"
    r"(?P<outcome_number>\d+))"
    r"\s*[:.)\-–—]?\s*(?P<description>.*)$",
    re.IGNORECASE,
)
_COURSE_NAME = re.compile(
    r"^\s*(?:course\s+(?:name|title)|subject\s+(?:name|title)|"
    r"name\s+of\s+(?:the\s+)?(?:course|subject))\s*[:\-]?\s*"
    r"(?P<value>.*?)\s*$",
    re.IGNORECASE,
)
_COURSE_CODE = re.compile(
    r"^\s*(?:course\s+(?:code|no\.?|number|id)|"
    r"subject\s+(?:code|no\.?|id)|paper\s+code)\s*[:\-]?\s*"
    r"(?P<value>[A-Z0-9][A-Z0-9 _/\-]{1,30}?)?\s*$",
    re.IGNORECASE,
)
_COURSE_CODE_VALUE = re.compile(
    r"^(?=[A-Z0-9/-]{3,20}$)(?=.*[A-Z])(?=.*\d)[A-Z0-9/-]+$",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^\s*(?:[•●▪◦*+\-\ufffd]\s*|\d+[.)]\s*)")
_UNIT_CONTEXT_END = re.compile(
    r"^\s*(?:course\s+outcomes?|CO\s*[-:]?\s*\d+|"
    r"textbooks?|references?|evaluation|assessment|"
    r"course\s+name|course\s+code)\b",
    re.IGNORECASE,
)
_OUTCOME_CONTEXT_END = re.compile(
    r"^\s*(?:list\s+of\s+experiments?|experiments?\s*\d*|"
    r"textbooks?|references?|unit\s*[-:]?\s*\d+|module\s*[-:]?\s*\d+)\b",
    re.IGNORECASE,
)
def extract_syllabus(pdf_path: str | Path) -> dict[str, Any]:
    """Extract page text and conservative syllabus fields from a PDF."""
    path = Path(pdf_path).expanduser()
    if not path.is_file():
        raise PDFExtractionError(f"PDF file does not exist: {path}")
    if path.suffix.casefold() != ".pdf":
        raise PDFExtractionError("The selected syllabus file must be a PDF.")

    try:
        document = pymupdf.open(path)
    except (pymupdf.FileDataError, OSError, RuntimeError) as error:
        raise PDFExtractionError(f"Could not open the syllabus PDF: {error}") from error

    try:
        with document:
            if document.is_encrypted:
                if not document.authenticate(""):
                    raise PDFExtractionError(
                        "This PDF is password-protected and cannot be read."
                    )
            if document.page_count == 0:
                raise PDFExtractionError("The PDF contains no pages.")

            pages = [
                {
                    "page": page_number,
                    "text": page.get_text("text", sort=True).strip(),
                }
                for page_number, page in enumerate(document, start=1)
            ]
            positioned_pages = [
                _positioned_lines(page) for page in document
            ]
            metadata = document.metadata or {}
    except PDFExtractionError:
        raise
    except (pymupdf.FileDataError, OSError, RuntimeError) as error:
        raise PDFExtractionError(
            f"Could not read the syllabus PDF: {error}"
        ) from error

    full_text = "\n".join(page["text"] for page in pages)
    if not full_text.strip():
        raise PDFExtractionError(
            "No selectable text was found. This may be a scanned PDF; "
            "scanned-document text recognition is not available."
        )

    course_name, course_code = _extract_course_identity(
        pages, metadata.get("title"), positioned_pages
    )
    units = _extract_units(pages)
    course_outcomes = _extract_course_outcomes(pages)
    warnings: list[str] = []
    if not course_name:
        warnings.append("Course or subject name was not clearly identified.")
    if not course_code:
        warnings.append("Course code was not clearly identified.")
    if not units:
        warnings.append("No clear unit headings were found; review the PDF manually.")
    if not course_outcomes:
        warnings.append(
            "No clearly labelled course outcomes were found; enter them manually."
        )
    if warnings:
        warnings.append(
            "Check all extracted details against the original syllabus before use."
        )

    return {
        "course_name": course_name,
        "course_code": course_code,
        "units": units,
        "course_outcomes": course_outcomes,
        "pages": pages,
        "source_file": str(path.resolve()),
        "review_status": "pending",
        "warnings": warnings,
    }


def validate_syllabus_data(data: dict[str, Any]) -> None:
    """Validate the editable syllabus structure before saving it."""
    if not isinstance(data, dict):
        raise ValueError("Syllabus data must be an object.")

    for field in ("course_name", "course_code"):
        value = data.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be text or empty.")

    for collection_name in ("units", "course_outcomes", "pages", "warnings"):
        if not isinstance(data.get(collection_name), list):
            raise ValueError(f"{collection_name} must be a list.")

    page_numbers: set[int] = set()
    for index, page in enumerate(data["pages"], start=1):
        if not isinstance(page, dict):
            raise ValueError(f"Page {index} must be an object.")
        if not isinstance(page.get("page"), int) or page["page"] < 1:
            raise ValueError(f"Page {index} needs a positive page number.")
        if not isinstance(page.get("text"), str):
            raise ValueError(f"Page {index} text must be text.")
        page_numbers.add(page["page"])

    for index, unit in enumerate(data["units"], start=1):
        if not isinstance(unit, dict):
            raise ValueError(f"Unit {index} must be an object.")
        if not isinstance(unit.get("unit_number"), int) or unit["unit_number"] < 1:
            raise ValueError(f"Unit {index} must have a positive unit number.")
        if not isinstance(unit.get("unit_title"), str):
            raise ValueError(f"Unit {index} title must be text.")
        if not isinstance(unit.get("topics"), list) or not all(
            isinstance(topic, str) for topic in unit["topics"]
        ):
            raise ValueError(f"Unit {index} topics must be a list of text.")
        _validate_pages(unit.get("source_pages"), f"Unit {index}", page_numbers)

    seen_outcomes: set[str] = set()
    for index, outcome in enumerate(data["course_outcomes"], start=1):
        if not isinstance(outcome, dict):
            raise ValueError(f"Course outcome {index} must be an object.")
        outcome_id = outcome.get("id")
        description = outcome.get("description")
        if not isinstance(outcome_id, str) or not outcome_id.strip():
            raise ValueError(f"Course outcome {index} needs an ID.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{outcome_id} needs a description.")
        normalized_id = outcome_id.strip().upper()
        if normalized_id in seen_outcomes:
            raise ValueError(f"Duplicate course outcome ID: {normalized_id}.")
        seen_outcomes.add(normalized_id)
        _validate_pages(outcome.get("source_pages"), normalized_id, page_numbers)

    if not isinstance(data.get("source_file"), str) or not data["source_file"].strip():
        raise ValueError("The original PDF path is required.")
    if data.get("review_status") not in {"pending", "reviewed"}:
        raise ValueError("review_status must be pending or reviewed.")
    if not all(isinstance(warning, str) for warning in data["warnings"]):
        raise ValueError("Warnings must contain only text.")


def save_syllabus_json(
    data: dict[str, Any], output_directory: str | Path | None = None
) -> Path:
    """Save reviewed extraction data without modifying the source PDF."""
    validate_syllabus_data(data)

    source_path = Path(data["source_file"])
    source_hash = hashlib.sha256(str(source_path.resolve()).encode("utf-8"))
    with source_path.open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            source_hash.update(chunk)
    source_digest = source_hash.hexdigest()[:12]
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source_path.stem).strip("_")
    if not safe_stem:
        safe_stem = "syllabus"

    if output_directory is None:
        output_directory = Path(__file__).resolve().parents[2] / "data" / "parsed_json"
    destination_directory = Path(output_directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / f"{safe_stem}-{source_digest}.json"
    temporary_path = destination.with_suffix(".json.tmp")
    try:
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(destination)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _extract_course_identity(
    pages: list[dict[str, Any]],
    metadata_title: str | None,
    positioned_pages: list[list[tuple[float, list[dict[str, Any]]]]],
) -> tuple[str | None, str | None]:
    course_name: str | None = None
    course_code: str | None = None

    for page in pages:
        lines = page["text"].splitlines()
        for line_index, line in enumerate(lines):
            line = _BULLET.sub("", line).strip()
            if course_name is None:
                name_match = _COURSE_NAME.match(line)
                if name_match:
                    value = name_match.group("value").strip(" :-\t")
                    if not value:
                        value = _next_labeled_value(lines, line_index)
                    if value and not _COURSE_CODE.match(value):
                        course_name = value
            if course_code is None:
                code_match = _COURSE_CODE.match(line)
                if code_match:
                    value = (code_match.group("value") or "").strip(" :-\t")
                    if not value:
                        value = _next_labeled_value(lines, line_index)
                    normalized_code = re.sub(r"\s+", "", value).upper()
                    if _COURSE_CODE_VALUE.fullmatch(normalized_code):
                        course_code = normalized_code
            if course_name and course_code:
                return course_name, course_code

    if course_name is None or course_code is None:
        layout_name, layout_code = _extract_tabular_course_identity(
            positioned_pages
        )
        course_name = course_name or layout_name
        course_code = course_code or layout_code

    if course_name is None and metadata_title:
        normalized_title = " ".join(metadata_title.split())
        normalized_title = re.sub(
            r"^(?:Microsoft\s+Word|Adobe\s+InDesign)\s*[-–—:]\s*",
            "",
            normalized_title,
            flags=re.IGNORECASE,
        )
        generic_metadata = (
            "untitled",
            "microsoft word",
            "document",
            "syllabus",
            "batch",
            "semester",
        )
        if (
            normalized_title
            and len(normalized_title) <= 100
            and not any(
                marker in normalized_title.casefold()
                for marker in generic_metadata
            )
        ):
            course_name = normalized_title

    return course_name, course_code


def _next_labeled_value(lines: list[str], label_index: int) -> str:
    for candidate in lines[label_index + 1 :]:
        value = " ".join(candidate.split()).strip(" :-\t")
        if not value:
            continue
        if (
            _COURSE_NAME.match(value)
            or _COURSE_CODE.match(value)
            or _is_metadata_heading(value)
        ):
            continue
        return value
    return ""


def _is_metadata_heading(value: str) -> bool:
    normalized = re.sub(r"[^a-z]", "", value.casefold())
    return normalized in {
        "coursecode",
        "coursename",
        "subjectcode",
        "subjectname",
        "papercode",
        "hours",
        "hoursperweek",
        "credits",
        "credit",
        "semester",
        "l",
        "t",
        "p",
    }


def _positioned_lines(
    page: pymupdf.Page,
) -> list[tuple[float, list[dict[str, Any]]]]:
    raw_lines = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            characters = [
                character
                for span in line.get("spans", [])
                for character in span.get("chars", [])
            ]
            if characters:
                raw_lines.append(
                    (float(line["bbox"][1]), characters)
                )

    positioned_lines: list[tuple[float, list[dict[str, Any]]]] = []
    for line_y, characters in sorted(raw_lines, key=lambda item: item[0]):
        matching_row = next(
            (
                index
                for index, (row_y, _) in enumerate(positioned_lines)
                if abs(row_y - line_y) <= 2
            ),
            None,
        )
        if matching_row is None:
            positioned_lines.append((line_y, characters))
        else:
            row_y, row_characters = positioned_lines[matching_row]
            positioned_lines[matching_row] = (
                row_y,
                row_characters + characters,
            )

    return [
        (line_y, sorted(characters, key=lambda item: item["bbox"][0]))
        for line_y, characters in positioned_lines
    ]


def _extract_tabular_course_identity(
    positioned_pages: list[list[tuple[float, list[dict[str, Any]]]]],
) -> tuple[str | None, str | None]:
    for lines in positioned_pages:
        code_headers: list[tuple[float, float]] = []
        name_headers: list[tuple[float, float]] = []
        later_columns: list[tuple[float, float]] = []
        for line_y, characters in lines:
            line_text = "".join(character["c"] for character in characters)
            folded_text = line_text.casefold()
            code_match = re.search(
                r"(?:course|subject)\s*(?:code|no\.?|number|id)|paper\s*code",
                folded_text,
            )
            name_match = re.search(
                r"(?:course|subject)\s*(?:name|title)|"
                r"name\s*of\s*(?:the\s*)?(?:course|subject)",
                folded_text,
            )
            later_match = re.search(
                r"hours|credits|semester|\bL\s*T\s*P\b",
                folded_text,
            )
            if code_match is not None:
                code_headers.append(
                    (
                        line_y,
                        float(characters[code_match.start()]["bbox"][0]),
                    )
                )
            if name_match is not None:
                name_headers.append(
                    (
                        line_y,
                        float(characters[name_match.start()]["bbox"][0]),
                    )
                )
            if later_match is not None:
                later_columns.append(
                    (
                        line_y,
                        float(characters[later_match.start()]["bbox"][0]),
                    )
                )

        header_pairs = [
            (code_header, name_header)
            for code_header in code_headers
            for name_header in name_headers
            if abs(code_header[0] - name_header[0]) <= 30
            and code_header[1] < name_header[1]
        ]
        if not header_pairs:
            continue
        code_header, name_header = min(
            header_pairs,
            key=lambda pair: (
                abs(pair[0][0] - pair[1][0]),
                pair[0][0] + pair[1][0],
            ),
        )
        header_y = max(code_header[0], name_header[0])
        next_column_x = next(
            (
                column_x
                for column_y, column_x in later_columns
                if abs(column_y - header_y) <= 30
                and column_x > name_header[1]
            ),
            None,
        )

        for line_y, characters in lines:
            if line_y <= header_y or line_y - header_y > 110:
                continue
            code_characters = [
                character
                for character in characters
                if code_header[1] - 4
                <= float(character["bbox"][0])
                < name_header[1] - 4
            ]
            code_text = "".join(
                character["c"] for character in code_characters
            )
            course_code = re.sub(r"[^A-Z0-9/-]", "", code_text.upper())
            if _COURSE_CODE_VALUE.fullmatch(course_code) is None:
                continue
            row_y = line_y
            name_characters = [
                character
                for name_line_y, name_line in lines
                if abs(name_line_y - row_y) <= 24
                for character in name_line
                if float(character["bbox"][0]) >= name_header[1] - 2
                and (
                    next_column_x is None
                    or float(character["bbox"][0]) < next_column_x
                )
            ]
            course_name = " ".join(
                "".join(character["c"] for character in name_characters).split()
            )
            return course_name or None, course_code

    return None, None


def _extract_units(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    current_unit: dict[str, Any] | None = None

    for page in pages:
        page_number = page["page"]
        for raw_line in page["text"].splitlines():
            line = " ".join(_BULLET.sub("", raw_line).split())
            if not line:
                continue

            heading = _UNIT_HEADING.match(line)
            if heading:
                number = _unit_number(heading.group("number"))
                if number is None:
                    current_unit = None
                    continue
                title = re.sub(
                    r"(?:^|\s)\d+\s*hrs?\b.*$",
                    "",
                    heading.group("title").strip(),
                    flags=re.IGNORECASE,
                )
                current_unit = {
                    "unit_number": number,
                    "unit_title": title,
                    "topics": [],
                    "source_pages": [page_number],
                }
                units.append(current_unit)
                continue

            if current_unit is None:
                continue
            if _UNIT_CONTEXT_END.match(line):
                current_unit = None
                continue

            topic = _BULLET.sub("", line).strip()
            if not topic:
                continue
            if re.match(r"^\d+\s*hrs?\b", topic, re.IGNORECASE):
                continue
            if not current_unit["unit_title"]:
                title_match = re.match(r"(?P<title>[^:;.]+)\s*:", topic)
                if title_match:
                    current_unit["unit_title"] = title_match.group("title").strip()
            if topic not in current_unit["topics"]:
                current_unit["topics"].append(topic)
            if page_number not in current_unit["source_pages"]:
                current_unit["source_pages"].append(page_number)

    return units


def _extract_course_outcomes(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    current_outcome: dict[str, Any] | None = None

    for page in pages:
        page_number = page["page"]
        for raw_line in page["text"].splitlines():
            line = " ".join(_BULLET.sub("", raw_line).split())
            if not line:
                continue

            match = _COURSE_OUTCOME.match(line)
            if match:
                number = match.group("co_number") or match.group("outcome_number")
                outcome_id = f"CO{int(number)}"
                current_outcome = next(
                    (item for item in outcomes if item["id"] == outcome_id),
                    None,
                )
                if current_outcome is None:
                    current_outcome = {
                        "id": outcome_id,
                        "description": "",
                        "source_pages": [],
                    }
                    outcomes.append(current_outcome)
                description = match.group("description").strip()
                if description:
                    current_outcome["description"] = _append_text(
                        current_outcome["description"], description
                    )
                if page_number not in current_outcome["source_pages"]:
                    current_outcome["source_pages"].append(page_number)
                continue

            if current_outcome is None:
                continue
            if _UNIT_HEADING.match(line) or _UNIT_CONTEXT_END.match(line):
                current_outcome = None
                continue
            if _OUTCOME_CONTEXT_END.match(line):
                current_outcome = None
                continue
            current_outcome["description"] = _append_text(
                current_outcome["description"], line
            )
            if page_number not in current_outcome["source_pages"]:
                current_outcome["source_pages"].append(page_number)

    return [
        outcome
        for outcome in outcomes
        if outcome["description"].strip()
    ]


def _append_text(existing: str, addition: str) -> str:
    return f"{existing} {addition}".strip()


def _validate_pages(value: Any, label: str, available_pages: set[int]) -> None:
    if not isinstance(value, list) or not all(
        isinstance(page, int) and page > 0 for page in value
    ):
        raise ValueError(f"{label} source_pages must contain positive page numbers.")
    if not set(value).issubset(available_pages):
        raise ValueError(f"{label} references a page not present in the PDF.")


def _unit_number(value: str) -> int | None:
    if value.isdigit():
        return int(value)

    total = 0
    previous_value = 0
    roman_values = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    for character in reversed(value.casefold()):
        current_value = roman_values.get(character)
        if current_value is None:
            return None
        if current_value < previous_value:
            total -= current_value
        else:
            total += current_value
            previous_value = current_value
    return total if total > 0 else None
