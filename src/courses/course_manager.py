from dataclasses import dataclass
from pathlib import Path


COURSE_OUTCOME_IDS = ("CO1", "CO2", "CO3", "CO4")


class CourseValidationError(ValueError):
    """Raised when course data is missing or conflicts with another course."""


@dataclass(frozen=True)
class Course:
    name: str
    code: str
    course_outcomes: dict[str, str]
    syllabus_pdf: str | None = None


class CourseManager:
    """Manage course records in memory until the SQLite phase is implemented."""

    def __init__(self) -> None:
        self._courses: dict[str, Course] = {}

    def list_courses(self) -> list[Course]:
        return sorted(
            self._courses.values(),
            key=lambda course: (course.name.casefold(), course.code),
        )

    def get_course(self, code: str) -> Course | None:
        normalized_code = self._normalize_code(code)
        return self._courses.get(normalized_code)

    def search(self, query: str) -> list[Course]:
        search_text = query.strip().casefold()
        if not search_text:
            return self.list_courses()

        return [
            course
            for course in self.list_courses()
            if search_text in course.name.casefold()
            or search_text in course.code.casefold()
        ]

    def add_course(
        self,
        name: str,
        code: str,
        course_outcomes: dict[str, str],
        syllabus_pdf: str | None = None,
    ) -> Course:
        course = self._validated_course(name, code, course_outcomes, syllabus_pdf)
        if course.code in self._courses:
            raise CourseValidationError(
                f"A course with code {course.code} already exists."
            )

        self._courses[course.code] = course
        return course

    def update_course(
        self,
        original_code: str,
        name: str,
        code: str,
        course_outcomes: dict[str, str],
        syllabus_pdf: str | None = None,
    ) -> Course:
        original_key = self._normalize_code(original_code)
        if original_key not in self._courses:
            raise CourseValidationError("The course to edit no longer exists.")

        course = self._validated_course(name, code, course_outcomes, syllabus_pdf)
        if course.code != original_key and course.code in self._courses:
            raise CourseValidationError(
                f"A course with code {course.code} already exists."
            )

        del self._courses[original_key]
        self._courses[course.code] = course
        return course

    def delete_course(self, code: str) -> bool:
        normalized_code = self._normalize_code(code)
        return self._courses.pop(normalized_code, None) is not None

    @classmethod
    def _validated_course(
        cls,
        name: str,
        code: str,
        course_outcomes: dict[str, str],
        syllabus_pdf: str | None,
    ) -> Course:
        normalized_name = name.strip()
        normalized_code = cls._normalize_code(code)
        if not normalized_name:
            raise CourseValidationError("Course name is required.")
        if not normalized_code:
            raise CourseValidationError("Course code is required.")

        if not isinstance(course_outcomes, dict):
            raise CourseValidationError("Enter descriptions for CO1 through CO4.")

        outcomes = {
            outcome_id: str(course_outcomes.get(outcome_id, "")).strip()
            for outcome_id in COURSE_OUTCOME_IDS
        }
        missing_outcomes = [
            outcome_id for outcome_id, description in outcomes.items() if not description
        ]
        if missing_outcomes:
            missing = ", ".join(missing_outcomes)
            raise CourseValidationError(
                f"Course outcome descriptions are required for: {missing}."
            )

        normalized_pdf = None
        if syllabus_pdf and syllabus_pdf.strip():
            path = Path(syllabus_pdf.strip())
            if path.suffix.casefold() != ".pdf":
                raise CourseValidationError("The syllabus file must be a PDF.")
            normalized_pdf = str(path)

        return Course(
            name=normalized_name,
            code=normalized_code,
            course_outcomes=outcomes,
            syllabus_pdf=normalized_pdf,
        )

    @staticmethod
    def _normalize_code(code: str) -> str:
        return code.strip().upper()
