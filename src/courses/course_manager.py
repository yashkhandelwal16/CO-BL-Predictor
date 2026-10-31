from dataclasses import dataclass
from pathlib import Path

from src.database.db_manager import (
    DatabaseError,
    DatabaseManager,
    DuplicateCourseCodeError,
)


COURSE_OUTCOME_IDS = ("CO1", "CO2", "CO3", "CO4", "CO5")


class CourseValidationError(ValueError):
    """Raised when course data is missing or conflicts with another course."""


@dataclass(frozen=True)
class Course:
    name: str
    code: str
    course_outcomes: dict[str, str]
    syllabus_pdf: str | None = None


class CourseManager:
    """Validate course data and persist it through the local database layer."""

    def __init__(self, database: DatabaseManager | None = None) -> None:
        self._owns_database = database is None
        self.database = database or DatabaseManager()

    def list_courses(self) -> list[Course]:
        return [self._as_course(record) for record in self.database.list_courses()]

    def get_course(self, code: str) -> Course | None:
        record = self.database.get_course(self._normalize_code(code))
        return self._as_course(record) if record is not None else None

    def search(self, query: str) -> list[Course]:
        return [self._as_course(record) for record in self.database.list_courses(query)]

    def add_course(
        self,
        name: str,
        code: str,
        course_outcomes: dict[str, str],
        syllabus_pdf: str | None = None,
    ) -> Course:
        course = self._validated_course(name, code, course_outcomes, syllabus_pdf)
        try:
            record = self.database.save_course(
                course.name,
                course.code,
                course.course_outcomes,
                course.syllabus_pdf,
            )
        except DuplicateCourseCodeError as error:
            raise CourseValidationError(str(error)) from error
        except DatabaseError as error:
            raise CourseValidationError(f"Course could not be saved: {error}") from error
        return self._as_course(record)

    def update_course(
        self,
        original_code: str,
        name: str,
        code: str,
        course_outcomes: dict[str, str],
        syllabus_pdf: str | None = None,
    ) -> Course:
        original_key = self._normalize_code(original_code)
        if self.database.get_course(original_key) is None:
            raise CourseValidationError("The course to edit no longer exists.")

        course = self._validated_course(name, code, course_outcomes, syllabus_pdf)
        try:
            record = self.database.save_course(
                course.name,
                course.code,
                course.course_outcomes,
                course.syllabus_pdf,
                original_code=original_key,
            )
        except DuplicateCourseCodeError as error:
            raise CourseValidationError(str(error)) from error
        except DatabaseError as error:
            raise CourseValidationError(f"Course could not be saved: {error}") from error
        return self._as_course(record)

    def save_reviewed_syllabus(
        self,
        syllabus_data: dict[str, object],
        parsed_json_path: str | Path,
        original_code: str | None = None,
    ) -> Course:
        name = str(syllabus_data.get("course_name") or "")
        code = str(syllabus_data.get("course_code") or "")
        source_file = str(syllabus_data.get("source_file") or "")
        raw_outcomes = syllabus_data.get("course_outcomes")
        if not isinstance(raw_outcomes, list):
            raise CourseValidationError("Reviewed course outcomes must be a list.")

        outcomes: dict[str, str] = {}
        for outcome in raw_outcomes:
            if not isinstance(outcome, dict):
                raise CourseValidationError("Each course outcome must be an object.")
            outcome_id = str(outcome.get("id", "")).strip().upper()
            description = str(outcome.get("description", "")).strip()
            if outcome_id in COURSE_OUTCOME_IDS and description:
                outcomes[outcome_id] = description

        course = self._validated_course(name, code, outcomes, source_file)
        try:
            existing = self.database.get_course(course.code)
            code_to_update = (
                self._normalize_code(original_code)
                if original_code
                else course.code if existing else None
            )
            record = self.database.save_course(
                course.name,
                course.code,
                course.course_outcomes,
                course.syllabus_pdf,
                original_code=code_to_update,
                reviewed_syllabus=syllabus_data,
                parsed_json_path=parsed_json_path,
            )
        except DuplicateCourseCodeError as error:
            raise CourseValidationError(str(error)) from error
        except DatabaseError as error:
            raise CourseValidationError(
                f"Course and reviewed syllabus could not be saved: {error}"
            ) from error
        return self._as_course(record)

    def delete_course(self, code: str) -> bool:
        try:
            return self.database.delete_course(self._normalize_code(code))
        except DatabaseError as error:
            raise CourseValidationError(f"Course could not be deleted: {error}") from error

    def close(self) -> None:
        if self._owns_database:
            self.database.close()

    @staticmethod
    def _as_course(record: dict[str, object]) -> Course:
        return Course(
            name=str(record["course_name"]),
            code=str(record["course_code"]),
            course_outcomes=dict(record["course_outcomes"]),
            syllabus_pdf=(
                str(record["syllabus_pdf"])
                if record["syllabus_pdf"] is not None
                else None
            ),
        )

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
            raise CourseValidationError(
                "Course outcomes must be provided as labeled descriptions."
            )
        invalid_outcomes = set(course_outcomes).difference(COURSE_OUTCOME_IDS)
        if invalid_outcomes:
            raise CourseValidationError(
                "Unsupported course outcome IDs: "
                + ", ".join(sorted(invalid_outcomes))
                + "."
            )
        outcomes = {
            outcome_id: str(course_outcomes.get(outcome_id) or "").strip()
            for outcome_id in COURSE_OUTCOME_IDS
            if str(course_outcomes.get(outcome_id) or "").strip()
        }

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
