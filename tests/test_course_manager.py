import pytest

from src.courses.course_manager import (
    COURSE_OUTCOME_IDS,
    CourseManager,
    CourseValidationError,
)


def outcomes(prefix: str = "Course") -> dict[str, str]:
    return {
        outcome_id: f"{prefix} outcome {index}"
        for index, outcome_id in enumerate(COURSE_OUTCOME_IDS, start=1)
    }


def test_courses_keep_independent_outcomes_and_syllabus_files(tmp_path) -> None:
    manager = CourseManager()
    first_pdf = tmp_path / "first.pdf"
    second_pdf = tmp_path / "second.pdf"

    first = manager.add_course(
        "First Course", "cs101", outcomes("First"), str(first_pdf)
    )
    second = manager.add_course(
        "Second Course", "CS202", outcomes("Second"), str(second_pdf)
    )

    manager.update_course(
        first.code,
        "First Course Updated",
        first.code,
        outcomes("Updated First"),
        str(first_pdf),
    )

    updated_first = manager.get_course("CS101")
    unchanged_second = manager.get_course("CS202")
    assert updated_first is not None
    assert unchanged_second is not None
    assert updated_first.course_outcomes["CO1"] == "Updated First outcome 1"
    assert unchanged_second.course_outcomes == second.course_outcomes
    assert updated_first.syllabus_pdf == str(first_pdf)
    assert unchanged_second.syllabus_pdf == str(second_pdf)


def test_course_search_matches_name_and_code() -> None:
    manager = CourseManager()
    manager.add_course("Data Structures", "CS301", outcomes())
    manager.add_course("Database Systems", "CS402", outcomes())

    assert [course.code for course in manager.search("struct")] == ["CS301"]
    assert [course.name for course in manager.search("cs402")] == [
        "Database Systems"
    ]


def test_course_code_is_case_insensitive_and_duplicates_are_rejected() -> None:
    manager = CourseManager()
    manager.add_course("First Course", "cs101", outcomes())

    with pytest.raises(CourseValidationError, match="CS101 already exists"):
        manager.add_course("Duplicate Course", "CS101", outcomes())


@pytest.mark.parametrize(
    ("name", "code", "course_outcomes", "message"),
    [
        (" ", "CS101", outcomes(), "Course name is required"),
        ("Course", " ", outcomes(), "Course code is required"),
        ("Course", "CS101", {"CO1": "Only one"}, "CO2"),
    ],
)
def test_required_course_fields_are_validated(
    name: str,
    code: str,
    course_outcomes: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(CourseValidationError, match=message):
        CourseManager().add_course(name, code, course_outcomes)


def test_only_pdf_syllabus_paths_are_accepted() -> None:
    with pytest.raises(CourseValidationError, match="must be a PDF"):
        CourseManager().add_course(
            "Course", "CS101", outcomes(), r"C:\syllabi\course.docx"
        )


def test_delete_course_reports_whether_a_record_existed() -> None:
    manager = CourseManager()
    manager.add_course("Course", "CS101", outcomes())

    assert manager.delete_course("cs101")
    assert not manager.delete_course("CS101")
