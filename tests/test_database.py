import json
import sqlite3

import pytest

from src.courses.course_manager import (
    COURSE_OUTCOME_IDS,
    CourseManager,
    CourseValidationError,
)
from src.database.db_manager import (
    DATABASE_SCHEMA_VERSION,
    DatabaseError,
    DatabaseManager,
    DuplicateCourseCodeError,
    default_database_path,
)


def outcome_descriptions(prefix: str = "Course") -> dict[str, str]:
    return {
        outcome_id: f"{prefix} description {number}"
        for number, outcome_id in enumerate(COURSE_OUTCOME_IDS, start=1)
    }


def reviewed_syllabus(pdf_path, course_name="Computer Networks"):
    return {
        "course_name": course_name,
        "course_code": "CS305",
        "units": [
            {
                "unit_number": 1,
                "unit_title": "Networking",
                "topics": ["Routing"],
                "source_pages": [1],
            }
        ],
        "course_outcomes": [
            {
                "id": outcome_id,
                "description": f"Reviewed {outcome_id}",
                "source_pages": [1],
            }
            for outcome_id in COURSE_OUTCOME_IDS
        ],
        "pages": [{"page": 1, "text": "Reviewed source text"}],
        "source_file": str(pdf_path),
        "review_status": "reviewed",
        "warnings": [],
    }


def test_schema_migration_creates_expected_tables_and_indexes(tmp_path) -> None:
    database = DatabaseManager(tmp_path / "application.sqlite3")

    with database._lock:
        version = database._connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in database._connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert version == DATABASE_SCHEMA_VERSION
    assert {
        "courses",
        "syllabus_versions",
        "course_outcomes",
        "questions",
        "predictions",
        "approved_labels",
    }.issubset(tables)
    assert database.integrity_check() == "ok"
    database.close()


def test_courses_and_outcomes_survive_database_restart(tmp_path) -> None:
    database_path = tmp_path / "application.sqlite3"
    manager = CourseManager(DatabaseManager(database_path))
    manager.add_course(
        "Computer Networks", "cs305", outcome_descriptions("Initial")
    )
    manager.database.close()

    restarted_manager = CourseManager(DatabaseManager(database_path))
    course = restarted_manager.get_course("CS305")

    assert course is not None
    assert course.name == "Computer Networks"
    assert course.course_outcomes == outcome_descriptions("Initial")
    assert restarted_manager.list_courses() == [course]
    restarted_manager.database.close()


def test_duplicate_codes_are_rejected_without_overwriting_existing_course() -> None:
    database = DatabaseManager()
    manager = CourseManager(database)
    manager.add_course("First Course", "CS101", outcome_descriptions("First"))

    with pytest.raises(CourseValidationError, match="CS101 already exists"):
        manager.add_course("Other Course", "cs101", outcome_descriptions("Other"))

    assert manager.get_course("CS101").name == "First Course"
    database.close()


def test_course_edit_preserves_syllabus_outcome_provenance(tmp_path) -> None:
    database = DatabaseManager(tmp_path / "application.sqlite3")
    manager = CourseManager(database)
    course = manager.add_course(
        "Computer Networks", "CS305", outcome_descriptions("Official")
    )
    pdf_path = tmp_path / "networks.pdf"
    pdf_path.write_bytes(b"course source")
    syllabus_data = reviewed_syllabus(pdf_path)
    database.save_syllabus_version(syllabus_data, tmp_path / "networks.json")

    reviewed_course = manager.get_course(course.code)
    manager.update_course(
        course.code,
        "Computer Networks and Protocols",
        course.code,
        reviewed_course.course_outcomes,
    )
    stored_outcomes = database.get_course("CS305")["course_outcomes"]
    with database._lock:
        provenance = database._connection.execute(
            "SELECT source_syllabus_version_id FROM course_outcomes "
            "WHERE outcome_id = 'CO1'"
        ).fetchone()[0]

    assert stored_outcomes["CO1"] == "Reviewed CO1"
    assert provenance is not None
    database.close()


def test_reviewed_syllabus_version_and_outcomes_survive_restart(tmp_path) -> None:
    database_path = tmp_path / "application.sqlite3"
    source_pdf = tmp_path / "network-syllabus.pdf"
    source_pdf.write_bytes(b"official pdf bytes")
    parsed_json = tmp_path / "network-syllabus.json"
    parsed_json.write_text("{}", encoding="utf-8")

    database = DatabaseManager(database_path)
    manager = CourseManager(database)
    manager.add_course(
        "Computer Networks", "CS305", outcome_descriptions("Manual")
    )
    version = database.save_syllabus_version(
        reviewed_syllabus(source_pdf),
        parsed_json,
    )
    database.close()

    restarted = DatabaseManager(database_path)
    stored_course = restarted.get_course("CS305")
    versions = restarted.list_syllabus_versions("CS305")

    assert stored_course is not None
    assert stored_course["syllabus_pdf"] == str(source_pdf)
    assert stored_course["course_outcomes"] == {
        outcome_id: f"Reviewed {outcome_id}" for outcome_id in COURSE_OUTCOME_IDS
    }
    assert versions[0]["id"] == version["id"]
    assert versions[0]["version_number"] == 1
    assert versions[0]["review_status"] == "reviewed"
    assert versions[0]["parsed_json_path"] == str(parsed_json)
    assert versions[0]["content"]["units"][0]["topics"] == ["Routing"]
    restarted.close()


def test_partial_or_unlinked_syllabus_is_saved_without_inventing_a_course(
    tmp_path,
) -> None:
    database = DatabaseManager()
    source_pdf = tmp_path / "partial.pdf"
    source_pdf.write_bytes(b"source")
    data = reviewed_syllabus(source_pdf)
    data["course_code"] = "NOCOURSE"
    data["course_outcomes"] = data["course_outcomes"][:2]

    version = database.save_syllabus_version(data, tmp_path / "partial.json")

    assert version["course_id"] is None
    assert database.get_course("NOCOURSE") is None
    assert database.list_syllabus_versions("NOCOURSE")[0]["course_code"] == "NOCOURSE"
    database.close()


def test_questions_predictions_and_approved_corrections_are_separate() -> None:
    database = DatabaseManager()
    CourseManager(database).add_course(
        "Computer Networks", "CS305", outcome_descriptions()
    )
    question = database.create_question(
        "CS305",
        "Compare two routing strategies.",
        marks=5,
        question_type="Short answer",
    )
    prediction = database.create_prediction(
        question["id"],
        "CO2",
        "BL4",
        provider_name="Local",
        model_name="test-model",
        model_version="test-version",
    )
    database.save_approved_labels(
        question["id"],
        "CO2",
        "BL4",
        prediction_id=prediction["id"],
        review_status="approved",
        reviewed_by="faculty",
    )
    database.save_approved_labels(
        question["id"],
        "CO3",
        "BL3",
        prediction_id=prediction["id"],
        review_status="corrected",
        reviewed_by="faculty",
        review_note="Correction after review",
    )

    history = database.list_question_history("routing")

    assert len(history) == 1
    assert history[0]["question_text"] == "Compare two routing strategies."
    assert history[0]["predicted_co"] == "CO2"
    assert history[0]["predicted_bl"] == "BL4"
    assert history[0]["approved_co"] == "CO3"
    assert history[0]["approved_bl"] == "BL3"
    assert history[0]["label_review_status"] == "corrected"

    with database._lock:
        prediction_count = database._connection.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]
        review_rows = database._connection.execute(
            "SELECT review_status FROM approved_labels "
            "WHERE question_id = ? ORDER BY created_at",
            (question["id"],),
        ).fetchall()
    assert prediction_count == 1
    assert {row[0] for row in review_rows} == {"approved", "corrected"}
    database.close()


def test_invalid_labels_and_cross_question_prediction_links_are_rejected() -> None:
    database = DatabaseManager()
    manager = CourseManager(database)
    manager.add_course("Networks", "CS301", outcome_descriptions())
    question_one = database.create_question("CS301", "Question one")
    question_two = database.create_question("CS301", "Question two")
    prediction = database.create_prediction(question_one["id"], "CO1", "BL1")

    with pytest.raises(ValueError, match="Invalid course outcome"):
        database.create_prediction(question_one["id"], "CO5", "BL1")
    with pytest.raises(ValueError, match="Invalid Bloom-style"):
        database.create_prediction(question_one["id"], "CO1", "BL6")
    with pytest.raises(DatabaseError):
        database.save_approved_labels(
            question_two["id"],
            "CO1",
            "BL1",
            prediction_id=prediction["id"],
        )
    database.close()


def test_parameterized_search_treats_sql_as_user_text() -> None:
    database = DatabaseManager()
    CourseManager(database).add_course("Networks", "CS301", outcome_descriptions())
    database.create_question("CS301", "List network layers.")

    assert database.list_question_history("' OR 1=1 --") == []
    assert database.list_courses("' OR 1=1 --") == []
    database.close()


def test_backup_can_be_opened_and_json_export_contains_all_records(tmp_path) -> None:
    database = DatabaseManager(tmp_path / "source.sqlite3")
    CourseManager(database).add_course(
        "Networks", "CS301", outcome_descriptions()
    )
    question = database.create_question("CS301", "Explain routing.")
    prediction = database.create_prediction(question["id"], "CO2", "BL2")
    database.save_approved_labels(
        question["id"],
        "CO2",
        "BL2",
        prediction_id=prediction["id"],
    )

    backup_path = database.backup(tmp_path / "backup.sqlite3")
    export_path = database.export_json(tmp_path / "export.json")

    backup = DatabaseManager(backup_path)
    assert backup.get_course("CS301")["course_name"] == "Networks"
    assert len(backup.list_question_history()) == 1
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert len(exported["courses"]) == 1
    assert len(exported["questions"]) == 1
    assert len(exported["predictions"]) == 1
    assert len(exported["approved_labels"]) == 1
    backup.close()
    database.close()


def test_backup_refuses_to_overwrite_the_live_database(tmp_path) -> None:
    database_path = tmp_path / "application.sqlite3"
    database = DatabaseManager(database_path)

    with pytest.raises(ValueError, match="different file"):
        database.backup(database_path)
    database.close()


def test_newer_unknown_database_schema_is_rejected(tmp_path) -> None:
    database_path = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA user_version = 999")
    connection.close()

    with pytest.raises(DatabaseError, match="newer application version"):
        DatabaseManager(database_path)


def test_default_database_location_uses_user_local_app_data(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))

    assert default_database_path() == (
        tmp_path / "LocalAppData" / "CO-BL-Predictor" / "predictor.sqlite3"
    )
