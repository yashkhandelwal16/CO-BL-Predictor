import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterator


DATABASE_SCHEMA_VERSION = 4
VALID_CO_LABELS = {"CO1", "CO2", "CO3", "CO4", "CO5", "N/A"}
VALID_BL_LABELS = {"BL0", "BL1", "BL2", "BL3", "BL4", "BL5", "N/A"}


class DatabaseError(RuntimeError):
    """Raised when persisted application data cannot be read or changed."""


class DuplicateCourseCodeError(DatabaseError):
    """Raised when a course code is already assigned to another course."""


def default_database_path() -> Path:
    """Return a writable, per-user data location outside the app install folder."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        data_directory = Path(local_app_data)
    elif os.name == "nt":
        data_directory = Path.home() / "AppData" / "Local"
    else:
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        data_directory = (
            Path(xdg_data_home)
            if xdg_data_home
            else Path.home() / ".local" / "share"
        )
    return data_directory / "CO-BL-Predictor" / "predictor.sqlite3"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return str(uuid.uuid4())


class DatabaseManager:
    """Transactional SQLite access for courses, syllabi, questions, and reviews."""

    def __init__(self, database_path: str | Path = ":memory:") -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            path = Path(self.database_path).expanduser()
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise DatabaseError(
                    f"Could not create the user data directory: {error}"
                ) from error
            self.database_path = str(path)

        try:
            self._connection = sqlite3.connect(
                self.database_path,
                timeout=10,
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 10000")
            if self.database_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._lock = RLock()
            self._migrate()
        except sqlite3.Error as error:
            raise DatabaseError(f"Could not initialize local data: {error}") from error

    @property
    def user_data_directory(self) -> Path | None:
        if self.database_path == ":memory:":
            return None
        return Path(self.database_path).parent

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN")
                yield self._connection
                self._connection.commit()
            except sqlite3.Error as error:
                self._connection.rollback()
                raise DatabaseError(f"Could not update local data: {error}") from error
            except Exception:
                self._connection.rollback()
                raise

    def _migrate(self) -> None:
        with self._lock:
            current_version = int(
                self._connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if current_version > DATABASE_SCHEMA_VERSION:
                raise DatabaseError(
                    "This data file was created by a newer application version."
                )
            if current_version == DATABASE_SCHEMA_VERSION:
                return
            if current_version not in {0, 1, 2, 3}:
                raise DatabaseError(
                    f"No migration is available from database version "
                    f"{current_version}."
                )

        with self._transaction() as connection:
            if current_version == 0:
                for statement in _SCHEMA.split(";"):
                    if statement.strip():
                        connection.execute(statement)
            else:
                if current_version == 1:
                    self._migrate_v1_to_v2(connection)
                if current_version in {1, 2}:
                    self._migrate_v2_to_v3(connection)
                self._migrate_v3_to_v4(connection)
            connection.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")

    @staticmethod
    def _migrate_v3_to_v4(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE predictions_v4 ("
            "id TEXT PRIMARY KEY, "
            "question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE, "
            "syllabus_version_id TEXT REFERENCES syllabus_versions(id) "
            "ON DELETE SET NULL, "
            "co_label TEXT NOT NULL CHECK (co_label IN "
            "('CO1', 'CO2', 'CO3', 'CO4', 'CO5', 'N/A')), "
            "bl_label TEXT NOT NULL CHECK (bl_label IN "
            "('BL0', 'BL1', 'BL2', 'BL3', 'BL4', 'BL5', 'N/A')), "
            "status TEXT NOT NULL CHECK (status IN "
            "('predicted', 'insufficient_context', 'failed')), "
            "provider_name TEXT, model_name TEXT, model_version TEXT, "
            "created_at TEXT NOT NULL, UNIQUE (id, question_id))"
        )
        connection.execute(
            "INSERT INTO predictions_v4 SELECT * FROM predictions"
        )
        connection.execute(
            "CREATE TABLE approved_labels_v4 ("
            "id TEXT PRIMARY KEY, "
            "question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE, "
            "prediction_id TEXT, "
            "co_label TEXT NOT NULL CHECK (co_label IN "
            "('CO1', 'CO2', 'CO3', 'CO4', 'CO5', 'N/A')), "
            "bl_label TEXT NOT NULL CHECK (bl_label IN "
            "('BL0', 'BL1', 'BL2', 'BL3', 'BL4', 'BL5', 'N/A')), "
            "review_status TEXT NOT NULL CHECK "
            "(review_status IN ('approved', 'corrected')), "
            "reviewed_by TEXT, review_note TEXT, created_at TEXT NOT NULL, "
            "FOREIGN KEY (prediction_id, question_id) "
            "REFERENCES predictions_v4(id, question_id) ON DELETE CASCADE)"
        )
        connection.execute(
            "INSERT INTO approved_labels_v4 SELECT * FROM approved_labels"
        )
        connection.execute("DROP TABLE approved_labels")
        connection.execute("DROP TABLE predictions")
        connection.execute("ALTER TABLE predictions_v4 RENAME TO predictions")
        connection.execute(
            "ALTER TABLE approved_labels_v4 RENAME TO approved_labels"
        )
        connection.execute(
            "CREATE INDEX idx_predictions_question_created "
            "ON predictions(question_id, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX idx_approved_question_created "
            "ON approved_labels(question_id, created_at DESC)"
        )

    @staticmethod
    def _migrate_v2_to_v3(connection: sqlite3.Connection) -> None:
        connection.execute("ALTER TABLE courses ADD COLUMN archived_at TEXT")
        syllabus_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(syllabus_versions)"
            ).fetchall()
        }
        required_columns = {
            "id",
            "course_id",
            "version_number",
            "course_name",
            "course_code",
            "source_pdf_path",
            "content_json",
            "review_status",
            "imported_at",
            "reviewed_at",
        }
        if not required_columns.issubset(syllabus_columns):
            return
        versions = connection.execute(
            "SELECT id, course_name, course_code, source_pdf_path, content_json, "
            "imported_at, reviewed_at FROM syllabus_versions "
            "WHERE course_id IS NULL AND review_status = 'reviewed' "
            "AND trim(COALESCE(course_name, '')) != '' "
            "AND trim(COALESCE(course_code, '')) != '' "
            "ORDER BY imported_at, version_number, id"
        ).fetchall()
        for version in versions:
            try:
                content = json.loads(version["content_json"])
            except json.JSONDecodeError:
                continue
            if not isinstance(content, dict):
                continue
            outcomes = content.get("course_outcomes", [])
            if not isinstance(outcomes, list):
                outcomes = []
            valid_outcomes = [
                outcome
                for outcome in outcomes
                if isinstance(outcome, dict)
                and str(outcome.get("id", "")).strip().upper()
                in {"CO1", "CO2", "CO3", "CO4", "CO5"}
                and str(outcome.get("description", "")).strip()
            ]
            course_code = version["course_code"].strip().upper()
            course = connection.execute(
                "SELECT id FROM courses "
                "WHERE course_code = ? COLLATE NOCASE AND archived_at IS NULL",
                (course_code,),
            ).fetchone()
            if course is None:
                if not valid_outcomes:
                    continue
                course_id = _new_id()
                connection.execute(
                    "INSERT INTO courses "
                    "(id, course_name, course_code, syllabus_pdf, created_at, "
                    "updated_at, archived_at) VALUES (?, ?, ?, ?, ?, ?, NULL)",
                    (
                        course_id,
                        str(version["course_name"]).strip(),
                        course_code,
                        version["source_pdf_path"],
                        version["imported_at"],
                        version["reviewed_at"] or version["imported_at"],
                    ),
                )
            else:
                course_id = course["id"]

            next_version = connection.execute(
                "SELECT COALESCE(MAX(version_number), 0) + 1 "
                "FROM syllabus_versions WHERE course_id = ?",
                (course_id,),
            ).fetchone()[0]
            connection.execute(
                "UPDATE syllabus_versions SET course_id = ?, course_code = ?, "
                "version_number = ? WHERE id = ?",
                (course_id, course_code, next_version, version["id"]),
            )
            now = version["reviewed_at"] or version["imported_at"]
            connection.execute(
                "UPDATE courses SET course_name = ?, syllabus_pdf = ?, "
                "updated_at = ? WHERE id = ?",
                (
                    str(version["course_name"]).strip(),
                    version["source_pdf_path"],
                    now,
                    course_id,
                ),
            )
            for outcome in valid_outcomes:
                outcome_id = str(outcome["id"]).strip().upper()
                connection.execute(
                    "INSERT INTO course_outcomes "
                    "(course_id, outcome_id, description, source_pages_json, "
                    "source_syllabus_version_id, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(course_id, outcome_id) DO UPDATE SET "
                    "description = excluded.description, "
                    "source_pages_json = excluded.source_pages_json, "
                    "source_syllabus_version_id = excluded.source_syllabus_version_id, "
                    "updated_at = excluded.updated_at",
                    (
                        course_id,
                        outcome_id,
                        str(outcome["description"]).strip(),
                        json.dumps(outcome.get("source_pages", [])),
                        version["id"],
                        now,
                    ),
                )

    @staticmethod
    def _migrate_v1_to_v2(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE course_outcomes_v2 ("
            "course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE, "
            "outcome_id TEXT NOT NULL CHECK (outcome_id IN "
            "('CO1', 'CO2', 'CO3', 'CO4', 'CO5')), "
            "description TEXT NOT NULL CHECK (length(trim(description)) > 0), "
            "source_pages_json TEXT NOT NULL DEFAULT '[]', "
            "source_syllabus_version_id TEXT REFERENCES syllabus_versions(id) "
            "ON DELETE SET NULL, updated_at TEXT NOT NULL, "
            "PRIMARY KEY (course_id, outcome_id))"
        )
        connection.execute(
            "INSERT INTO course_outcomes_v2 SELECT * FROM course_outcomes"
        )
        connection.execute(
            "CREATE TABLE predictions_v2 ("
            "id TEXT PRIMARY KEY, "
            "question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE, "
            "syllabus_version_id TEXT REFERENCES syllabus_versions(id) "
            "ON DELETE SET NULL, "
            "co_label TEXT NOT NULL CHECK (co_label IN "
            "('CO1', 'CO2', 'CO3', 'CO4', 'CO5', 'N/A')), "
            "bl_label TEXT NOT NULL CHECK (bl_label IN "
            "('BL0', 'BL1', 'BL2', 'BL3', 'BL4', 'BL5', 'N/A')), "
            "status TEXT NOT NULL CHECK (status IN "
            "('predicted', 'insufficient_context', 'failed')), "
            "provider_name TEXT, model_name TEXT, model_version TEXT, "
            "created_at TEXT NOT NULL, UNIQUE (id, question_id))"
        )
        connection.execute(
            "INSERT INTO predictions_v2 SELECT * FROM predictions"
        )
        connection.execute(
            "CREATE TABLE approved_labels_v2 ("
            "id TEXT PRIMARY KEY, "
            "question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE, "
            "prediction_id TEXT, "
            "co_label TEXT NOT NULL CHECK (co_label IN "
            "('CO1', 'CO2', 'CO3', 'CO4', 'CO5', 'N/A')), "
            "bl_label TEXT NOT NULL CHECK (bl_label IN "
            "('BL0', 'BL1', 'BL2', 'BL3', 'BL4', 'BL5', 'N/A')), "
            "review_status TEXT NOT NULL CHECK "
            "(review_status IN ('approved', 'corrected')), "
            "reviewed_by TEXT, review_note TEXT, created_at TEXT NOT NULL, "
            "FOREIGN KEY (prediction_id, question_id) "
            "REFERENCES predictions_v2(id, question_id) ON DELETE CASCADE)"
        )
        connection.execute(
            "INSERT INTO approved_labels_v2 SELECT * FROM approved_labels"
        )

        for index_name in (
            "idx_approved_question_created",
            "idx_predictions_question_created",
        ):
            connection.execute(f"DROP INDEX IF EXISTS {index_name}")
        connection.execute("DROP TABLE approved_labels")
        connection.execute("DROP TABLE predictions")
        connection.execute("DROP TABLE course_outcomes")
        connection.execute("ALTER TABLE course_outcomes_v2 RENAME TO course_outcomes")
        connection.execute("ALTER TABLE predictions_v2 RENAME TO predictions")
        connection.execute("ALTER TABLE approved_labels_v2 RENAME TO approved_labels")
        connection.execute(
            "CREATE INDEX idx_predictions_question_created "
            "ON predictions(question_id, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX idx_approved_question_created "
            "ON approved_labels(question_id, created_at DESC)"
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def save_course(
        self,
        name: str,
        code: str,
        outcomes: dict[str, str],
        syllabus_pdf: str | None = None,
        original_code: str | None = None,
        *,
        reviewed_syllabus: dict[str, Any] | None = None,
        parsed_json_path: str | Path | None = None,
    ) -> dict[str, Any]:
        normalized_name = name.strip()
        normalized_code = code.strip().upper()
        normalized_original_code = (
            original_code.strip().upper() if original_code else None
        )
        if not normalized_name or not normalized_code:
            raise ValueError("Course name and code are required.")
        if not isinstance(outcomes, dict):
            raise ValueError("Course outcomes must be provided as labeled text.")
        valid_outcomes = {"CO1", "CO2", "CO3", "CO4", "CO5"}
        invalid_outcomes = set(outcomes).difference(valid_outcomes)
        if invalid_outcomes:
            raise ValueError(
                "Unsupported course outcome IDs: "
                + ", ".join(sorted(invalid_outcomes))
                + "."
            )
        normalized_outcomes = {
            label: str(description).strip()
            for label, description in outcomes.items()
            if str(description).strip()
        }
        if len(normalized_outcomes) != len(outcomes):
            raise ValueError("Course outcome descriptions cannot be empty.")

        if reviewed_syllabus is not None:
            if reviewed_syllabus.get("review_status") != "reviewed":
                raise ValueError("Only a reviewed syllabus can be saved as a course.")
            if (reviewed_syllabus.get("course_name") or "").strip().casefold() != (
                normalized_name.casefold()
            ):
                raise ValueError("The reviewed syllabus course name does not match.")
            if (reviewed_syllabus.get("course_code") or "").strip().upper() != (
                normalized_code
            ):
                raise ValueError("The reviewed syllabus course code does not match.")
            source_file = reviewed_syllabus.get("source_file")
            if not isinstance(source_file, str) or not source_file.strip():
                raise ValueError("The reviewed syllabus source PDF is required.")
            syllabus_pdf = source_file

        syllabus_path = str(Path(syllabus_pdf).expanduser()) if syllabus_pdf else None
        now = _timestamp()
        with self._transaction() as connection:
            existing = None
            if normalized_original_code:
                existing = connection.execute(
                    "SELECT id, created_at, syllabus_pdf, archived_at FROM courses "
                    "WHERE course_code = ? COLLATE NOCASE",
                    (normalized_original_code,),
                ).fetchone()
                if existing is None or existing["archived_at"] is not None:
                    raise DatabaseError("The course to edit no longer exists.")
            else:
                existing = connection.execute(
                    "SELECT id, created_at, syllabus_pdf, archived_at FROM courses "
                    "WHERE course_code = ? COLLATE NOCASE",
                    (normalized_code,),
                ).fetchone()
                if existing is not None and existing["archived_at"] is None:
                    raise DuplicateCourseCodeError(
                        f"A course with code {normalized_code} already exists."
                    )

            if existing is None:
                course_id = _new_id()
                connection.execute(
                    "INSERT INTO courses "
                    "(id, course_name, course_code, syllabus_pdf, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        course_id,
                        normalized_name,
                        normalized_code,
                        syllabus_path,
                        now,
                        now,
                    ),
                )
                created_at = now
            else:
                course_id = existing["id"]
                created_at = existing["created_at"]
                other_course = connection.execute(
                    "SELECT id FROM courses WHERE course_code = ? COLLATE NOCASE "
                    "AND id != ?",
                    (normalized_code, course_id),
                ).fetchone()
                if other_course is not None:
                    raise DuplicateCourseCodeError(
                        f"A course with code {normalized_code} already exists."
                    )

                connection.execute(
                    "UPDATE courses SET course_name = ?, course_code = ?, "
                    "syllabus_pdf = ?, updated_at = ?, archived_at = NULL "
                    "WHERE id = ?",
                    (
                        normalized_name,
                        normalized_code,
                        syllabus_path,
                        now,
                        course_id,
                    ),
                )

            for outcome_id, description in normalized_outcomes.items():
                connection.execute(
                    "INSERT INTO course_outcomes "
                    "(course_id, outcome_id, description, source_pages_json, "
                    "source_syllabus_version_id, updated_at) "
                    "VALUES (?, ?, ?, '[]', NULL, ?) "
                    "ON CONFLICT(course_id, outcome_id) DO UPDATE SET "
                    "description = excluded.description, "
                    "source_pages_json = CASE "
                    "WHEN course_outcomes.description != excluded.description "
                    "THEN '[]' ELSE course_outcomes.source_pages_json END, "
                    "source_syllabus_version_id = CASE "
                    "WHEN course_outcomes.description != excluded.description "
                    "THEN NULL ELSE course_outcomes.source_syllabus_version_id END, "
                    "updated_at = excluded.updated_at",
                    (course_id, outcome_id, str(description).strip(), now),
                )

            if normalized_outcomes:
                placeholders = ", ".join("?" for _ in normalized_outcomes)
                connection.execute(
                    "DELETE FROM course_outcomes WHERE course_id = ? "
                    f"AND outcome_id NOT IN ({placeholders})",
                    (course_id, *normalized_outcomes),
                )
            else:
                connection.execute(
                    "DELETE FROM course_outcomes WHERE course_id = ?",
                    (course_id,),
                )

            if reviewed_syllabus is not None:
                reviewed_version = self._insert_syllabus_version(
                    connection,
                    course_id=course_id,
                    course_name=normalized_name,
                    course_code=normalized_code,
                    source_pdf_path=syllabus_path,
                    source_sha256=_file_sha256(syllabus_path),
                    parsed_json_path=(
                        str(Path(parsed_json_path).expanduser())
                        if parsed_json_path
                        else None
                    ),
                    content=reviewed_syllabus,
                    review_status="reviewed",
                    now=now,
                )
                self._save_reviewed_outcomes(
                    connection,
                    course_id,
                    reviewed_version["id"],
                    reviewed_syllabus.get("course_outcomes", []),
                    now,
                )

            current_syllabus = (
                connection.execute(
                    "SELECT source_pdf_path, source_sha256 FROM syllabus_versions "
                    "WHERE course_id = ? ORDER BY version_number DESC LIMIT 1",
                    (course_id,),
                ).fetchone()
                if syllabus_path
                else None
            )
            syllabus_digest = _file_sha256(syllabus_path) if syllabus_path else None
            if reviewed_syllabus is None and syllabus_path and (
                current_syllabus is None
                or syllabus_path != current_syllabus["source_pdf_path"]
                or syllabus_digest != current_syllabus["source_sha256"]
            ):
                self._insert_syllabus_version(
                    connection,
                    course_id=course_id,
                    course_name=normalized_name,
                    course_code=normalized_code,
                    source_pdf_path=syllabus_path,
                    source_sha256=syllabus_digest,
                    parsed_json_path=None,
                    content={
                        "course_name": normalized_name,
                        "course_code": normalized_code,
                        "course_outcomes": [
                            {"id": outcome_id, "description": description.strip()}
                            for outcome_id, description in normalized_outcomes.items()
                        ],
                    },
                    review_status="pending",
                    now=now,
                )

            return {
                "id": course_id,
                "course_name": normalized_name,
                "course_code": normalized_code,
                "syllabus_pdf": syllabus_path,
                "created_at": created_at,
                "updated_at": now,
                "course_outcomes": dict(normalized_outcomes),
            }

    def list_courses(self, query: str = "") -> list[dict[str, Any]]:
        search_text = query.strip()
        with self._lock:
            try:
                if search_text:
                    pattern = f"%{search_text}%"
                    rows = self._connection.execute(
                        "SELECT c.id, c.course_name, c.course_code, c.syllabus_pdf, "
                        "c.created_at, c.updated_at, c.archived_at FROM courses c "
                        "WHERE c.archived_at IS NULL AND ("
                        "c.course_name LIKE ? COLLATE NOCASE "
                        "OR c.course_code LIKE ? COLLATE NOCASE) "
                        "ORDER BY c.course_name COLLATE NOCASE, c.course_code",
                        (pattern, pattern),
                    ).fetchall()
                else:
                    rows = self._connection.execute(
                        "SELECT id, course_name, course_code, syllabus_pdf, "
                        "created_at, updated_at, archived_at FROM courses "
                        "WHERE archived_at IS NULL "
                        "ORDER BY course_name COLLATE NOCASE, course_code"
                    ).fetchall()
                return [
                    self._course_from_row(self._connection, row) for row in rows
                ]
            except sqlite3.Error as error:
                raise DatabaseError(f"Could not read courses: {error}") from error

    def get_course(self, code: str) -> dict[str, Any] | None:
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT id, course_name, course_code, syllabus_pdf, "
                    "created_at, updated_at, archived_at FROM courses "
                    "WHERE course_code = ? COLLATE NOCASE AND archived_at IS NULL",
                    (code.strip(),),
                ).fetchone()
                return self._course_from_row(self._connection, row) if row else None
            except sqlite3.Error as error:
                raise DatabaseError(f"Could not read the course: {error}") from error

    def get_course_by_id(self, course_id: str) -> dict[str, Any] | None:
        with self._lock:
            try:
                row = self._connection.execute(
                    "SELECT id, course_name, course_code, syllabus_pdf, "
                    "created_at, updated_at, archived_at FROM courses "
                    "WHERE id = ? AND archived_at IS NULL",
                    (course_id,),
                ).fetchone()
                return self._course_from_row(self._connection, row) if row else None
            except sqlite3.Error as error:
                raise DatabaseError(f"Could not read the course: {error}") from error

    @staticmethod
    def _course_from_row(
        connection: sqlite3.Connection, row: sqlite3.Row
    ) -> dict[str, Any]:
        outcomes = {
            outcome["outcome_id"]: outcome["description"]
            for outcome in connection.execute(
                "SELECT outcome_id, description FROM course_outcomes "
                "WHERE course_id = ? ORDER BY outcome_id",
                (row["id"],),
            ).fetchall()
        }
        return {
            "id": row["id"],
            "course_name": row["course_name"],
            "course_code": row["course_code"],
            "syllabus_pdf": row["syllabus_pdf"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "archived_at": row["archived_at"] if "archived_at" in row.keys() else None,
            "course_outcomes": outcomes,
        }

    def delete_course(self, code: str) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE courses SET archived_at = ?, updated_at = ? "
                "WHERE course_code = ? COLLATE NOCASE AND archived_at IS NULL",
                (_timestamp(), _timestamp(), code.strip()),
            )
            return cursor.rowcount > 0

    def save_syllabus_version(
        self, data: dict[str, Any], parsed_json_path: str | Path
    ) -> dict[str, Any]:
        source_pdf_path = str(Path(data["source_file"]).expanduser())
        course_name = (data.get("course_name") or "").strip() or None
        course_code = (data.get("course_code") or "").strip().upper() or None
        review_status = data.get("review_status", "pending")
        if review_status not in {"pending", "reviewed"}:
            raise ValueError("Syllabus review status must be pending or reviewed.")

        now = _timestamp()
        content_json = json.dumps(data, ensure_ascii=False, sort_keys=True)
        with self._transaction() as connection:
            course = None
            if course_code:
                course = connection.execute(
                    "SELECT id, course_name FROM courses "
                    "WHERE course_code = ? COLLATE NOCASE AND archived_at IS NULL",
                    (course_code,),
                ).fetchone()
            course_id = course["id"] if course else None
            if course and course_name and course["course_name"].casefold() != course_name.casefold():
                content = dict(data)
                warnings = list(content.get("warnings", []))
                warnings.append(
                    "The extracted course name differs from the saved course with "
                    "this code; verify the association."
                )
                content["warnings"] = warnings
                content_json = json.dumps(content, ensure_ascii=False, sort_keys=True)

            source_version = self._insert_syllabus_version(
                connection,
                course_id=course_id,
                course_name=course_name,
                course_code=course_code,
                source_pdf_path=source_pdf_path,
                source_sha256=_file_sha256(source_pdf_path),
                parsed_json_path=str(parsed_json_path),
                content=data,
                review_status=review_status,
                now=now,
            )

            if course_id and review_status == "reviewed":
                self._save_reviewed_outcomes(
                    connection,
                    course_id,
                    source_version["id"],
                    data.get("course_outcomes", []),
                    now,
                )
                connection.execute(
                    "UPDATE courses SET syllabus_pdf = ?, updated_at = ? WHERE id = ?",
                    (source_pdf_path, now, course_id),
                )
            return source_version

    @staticmethod
    def _insert_syllabus_version(
        connection: sqlite3.Connection,
        *,
        course_id: str | None,
        course_name: str | None,
        course_code: str | None,
        source_pdf_path: str,
        source_sha256: str | None,
        parsed_json_path: str | None,
        content: dict[str, Any],
        review_status: str,
        now: str,
    ) -> dict[str, Any]:
        row = connection.execute(
            "SELECT COALESCE(MAX(version_number), 0) + 1 "
            "FROM syllabus_versions WHERE course_id IS ? AND course_code IS ?",
            (course_id, course_code),
        ).fetchone()
        version_number = int(row[0])
        version_id = _new_id()
        reviewed_at = now if review_status == "reviewed" else None
        connection.execute(
            "INSERT INTO syllabus_versions "
            "(id, course_id, version_number, course_name, course_code, "
            "source_pdf_path, source_sha256, parsed_json_path, content_json, "
            "review_status, imported_at, reviewed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                version_id,
                course_id,
                version_number,
                course_name,
                course_code,
                source_pdf_path,
                source_sha256,
                parsed_json_path,
                json.dumps(content, ensure_ascii=False, sort_keys=True),
                review_status,
                now,
                reviewed_at,
            ),
        )
        return {
            "id": version_id,
            "course_id": course_id,
            "version_number": version_number,
            "course_name": course_name,
            "course_code": course_code,
            "source_pdf_path": source_pdf_path,
            "source_sha256": source_sha256,
            "parsed_json_path": parsed_json_path,
            "review_status": review_status,
            "imported_at": now,
            "reviewed_at": reviewed_at,
        }

    @staticmethod
    def _save_reviewed_outcomes(
        connection: sqlite3.Connection,
        course_id: str,
        syllabus_version_id: str,
        outcomes: list[dict[str, Any]],
        now: str,
    ) -> None:
        by_id = {
            str(outcome.get("id", "")).strip().upper(): outcome
            for outcome in outcomes
        }
        for outcome_id, outcome in by_id.items():
            if outcome_id not in {"CO1", "CO2", "CO3", "CO4", "CO5"}:
                continue
            if not str(outcome.get("description", "")).strip():
                continue
            connection.execute(
                "INSERT INTO course_outcomes "
                "(course_id, outcome_id, description, source_pages_json, "
                "source_syllabus_version_id, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(course_id, outcome_id) DO UPDATE SET "
                "description = excluded.description, "
                "source_pages_json = excluded.source_pages_json, "
                "source_syllabus_version_id = excluded.source_syllabus_version_id, "
                "updated_at = excluded.updated_at",
                (
                    course_id,
                    outcome_id,
                    outcome["description"].strip(),
                    json.dumps(outcome.get("source_pages", [])),
                    syllabus_version_id,
                    now,
                ),
            )

    def list_syllabus_versions(
        self,
        course_code: str | None = None,
        *,
        course_id: str | None = None,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT id, course_id, version_number, course_name, course_code, "
            "source_pdf_path, source_sha256, parsed_json_path, content_json, "
            "review_status, imported_at, reviewed_at "
            "FROM syllabus_versions"
        )
        parameters: tuple[Any, ...] = ()
        if course_id:
            query += " WHERE course_id = ?"
            parameters = (course_id,)
        elif course_code:
            query += " WHERE course_code = ? COLLATE NOCASE"
            parameters = (course_code.strip(),)
        query += " ORDER BY imported_at DESC, version_number DESC"
        with self._lock:
            try:
                rows = self._connection.execute(query, parameters).fetchall()
                return [
                    {
                        **dict(row),
                        "content": json.loads(row["content_json"]),
                    }
                    for row in rows
                ]
            except (sqlite3.Error, json.JSONDecodeError) as error:
                raise DatabaseError(
                    f"Could not read syllabus history: {error}"
                ) from error

    def create_question(
        self,
        course_id_or_code: str,
        question_text: str,
        marks: float | None = None,
        question_type: str | None = None,
    ) -> dict[str, Any]:
        text = question_text.strip()
        if not text:
            raise ValueError("Question text is required.")
        if marks is not None and marks < 0:
            raise ValueError("Question marks cannot be negative.")
        now = _timestamp()
        question_id = _new_id()
        with self._transaction() as connection:
            course = connection.execute(
                "SELECT id, course_code FROM courses "
                "WHERE (id = ? OR course_code = ? COLLATE NOCASE) "
                "AND archived_at IS NULL",
                (course_id_or_code.strip(), course_id_or_code.strip()),
            ).fetchone()
            if course is None:
                raise DatabaseError("The selected course no longer exists.")
            connection.execute(
                "INSERT INTO questions "
                "(id, course_id, question_text, marks, question_type, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    question_id,
                    course["id"],
                    text,
                    marks,
                    question_type.strip() if question_type else None,
                    now,
                    now,
                ),
            )
        return {
            "id": question_id,
            "course_id": course["id"],
            "course_code": course["course_code"],
            "question_text": text,
            "marks": marks,
            "question_type": question_type,
            "created_at": now,
            "updated_at": now,
        }

    def create_prediction(
        self,
        question_id: str,
        co_label: str,
        bl_label: str,
        *,
        status: str = "predicted",
        provider_name: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        syllabus_version_id: str | None = None,
    ) -> dict[str, Any]:
        if co_label not in VALID_CO_LABELS:
            raise ValueError(f"Invalid course outcome label: {co_label}.")
        if bl_label not in VALID_BL_LABELS:
            raise ValueError(f"Invalid Bloom-style label: {bl_label}.")
        if status not in {"predicted", "insufficient_context", "failed"}:
            raise ValueError("Invalid prediction status.")
        now = _timestamp()
        prediction_id = _new_id()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO predictions "
                "(id, question_id, syllabus_version_id, co_label, bl_label, status, "
                "provider_name, model_name, model_version, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    prediction_id,
                    question_id,
                    syllabus_version_id,
                    co_label,
                    bl_label,
                    status,
                    provider_name,
                    model_name,
                    model_version,
                    now,
                ),
            )
        return {
            "id": prediction_id,
            "question_id": question_id,
            "syllabus_version_id": syllabus_version_id,
            "co_label": co_label,
            "bl_label": bl_label,
            "status": status,
            "provider_name": provider_name,
            "model_name": model_name,
            "model_version": model_version,
            "created_at": now,
        }

    def save_approved_labels(
        self,
        question_id: str,
        co_label: str,
        bl_label: str,
        *,
        prediction_id: str | None = None,
        review_status: str = "approved",
        reviewed_by: str | None = None,
        review_note: str | None = None,
    ) -> dict[str, Any]:
        if co_label not in VALID_CO_LABELS:
            raise ValueError(f"Invalid approved course outcome label: {co_label}.")
        if bl_label not in VALID_BL_LABELS:
            raise ValueError(f"Invalid approved Bloom-style label: {bl_label}.")
        if review_status not in {"approved", "corrected"}:
            raise ValueError("Review status must be approved or corrected.")

        now = _timestamp()
        review_id = _new_id()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO approved_labels "
                "(id, question_id, prediction_id, co_label, bl_label, review_status, "
                "reviewed_by, review_note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    review_id,
                    question_id,
                    prediction_id,
                    co_label,
                    bl_label,
                    review_status,
                    reviewed_by.strip() if reviewed_by else None,
                    review_note.strip() if review_note else None,
                    now,
                ),
            )
        return {
            "id": review_id,
            "question_id": question_id,
            "prediction_id": prediction_id,
            "co_label": co_label,
            "bl_label": bl_label,
            "review_status": review_status,
            "reviewed_by": reviewed_by,
            "review_note": review_note,
            "created_at": now,
        }

    def list_question_history(
        self, query: str = "", course_code: str | None = None
    ) -> list[dict[str, Any]]:
        statement = """
            SELECT q.id AS question_id, q.course_id, q.question_text,
                   q.marks, q.question_type,
                   q.created_at, c.course_name, c.course_code,
                   p.id AS prediction_id,
                   p.co_label AS predicted_co, p.bl_label AS predicted_bl,
                   p.model_name, p.model_version, p.provider_name,
                   a.co_label AS approved_co, a.bl_label AS approved_bl,
                   a.review_status AS label_review_status
            FROM questions q
            JOIN courses c ON c.id = q.course_id
            LEFT JOIN predictions p ON p.id = (
                SELECT p2.id FROM predictions p2
                WHERE p2.question_id = q.id
                ORDER BY p2.created_at DESC, p2.id DESC LIMIT 1
            )
            LEFT JOIN approved_labels a ON a.id = (
                SELECT a2.id FROM approved_labels a2
                WHERE a2.question_id = q.id
                ORDER BY a2.created_at DESC, a2.rowid DESC LIMIT 1
            )
            WHERE (? = '' OR q.question_text LIKE ? COLLATE NOCASE
                   OR c.course_name LIKE ? COLLATE NOCASE
                   OR c.course_code LIKE ? COLLATE NOCASE)
              AND (? IS NULL OR c.course_code = ? COLLATE NOCASE)
            ORDER BY q.created_at DESC, q.id DESC
        """
        search = query.strip()
        pattern = f"%{search}%"
        with self._lock:
            try:
                rows = self._connection.execute(
                    statement,
                    (
                        search,
                        pattern,
                        pattern,
                        pattern,
                        course_code,
                        course_code,
                    ),
                ).fetchall()
                return [dict(row) for row in rows]
            except sqlite3.Error as error:
                raise DatabaseError(f"Could not read question history: {error}") from error

    def backup(self, destination: str | Path) -> Path:
        destination_path = Path(destination).expanduser()
        if self.database_path != ":memory:":
            source_path = Path(self.database_path).resolve()
            if destination_path.resolve() == source_path:
                raise ValueError("Choose a different file for the backup.")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination_path.with_name(
            f".{destination_path.name}.{uuid.uuid4().hex}.tmp"
        )
        with self._lock:
            backup_connection: sqlite3.Connection | None = None
            try:
                backup_connection = sqlite3.connect(temporary_path)
                self._connection.backup(backup_connection)
                backup_connection.close()
                backup_connection = None
                temporary_path.replace(destination_path)
            except (sqlite3.Error, OSError) as error:
                cleanup_error = None
                if backup_connection is not None:
                    backup_connection.close()
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError as cleanup_exception:
                    cleanup_error = cleanup_exception
                message = f"Could not create a backup: {error}"
                if cleanup_error:
                    message += f" Temporary file cleanup also failed: {cleanup_error}"
                raise DatabaseError(message) from error
        return destination_path

    def export_json(self, destination: str | Path) -> Path:
        destination_path = Path(destination).expanduser()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        export = {
            "exported_at": _timestamp(),
            "schema_version": DATABASE_SCHEMA_VERSION,
            "courses": [],
            "syllabus_versions": [],
            "questions": [],
            "predictions": [],
            "approved_labels": [],
        }
        with self._lock:
            try:
                export["courses"] = [
                    self._course_from_row(self._connection, row)
                    for row in self._connection.execute(
                        "SELECT id, course_name, course_code, syllabus_pdf, "
                        "created_at, updated_at, archived_at FROM courses "
                        "ORDER BY course_code"
                    ).fetchall()
                ]
                for course in export["courses"]:
                    course["course_outcomes"] = {
                        row["outcome_id"]: row["description"]
                        for row in self._connection.execute(
                            "SELECT outcome_id, description FROM course_outcomes "
                            "WHERE course_id = ? ORDER BY outcome_id",
                            (course["id"],),
                        ).fetchall()
                    }
                for key, table in (
                    ("syllabus_versions", "syllabus_versions"),
                    ("questions", "questions"),
                    ("predictions", "predictions"),
                    ("approved_labels", "approved_labels"),
                ):
                    rows = self._connection.execute(
                        f"SELECT * FROM {table} ORDER BY rowid"
                    ).fetchall()
                    export[key] = [dict(row) for row in rows]
                for version in export["syllabus_versions"]:
                    version["content"] = json.loads(version.pop("content_json"))

                _write_json_atomically(destination_path, export)
            except (sqlite3.Error, json.JSONDecodeError, OSError) as error:
                raise DatabaseError(f"Could not export local data: {error}") from error
        return destination_path

    def integrity_check(self) -> str:
        with self._lock:
            try:
                return str(self._connection.execute("PRAGMA integrity_check").fetchone()[0])
            except sqlite3.Error as error:
                raise DatabaseError(f"Could not verify local data: {error}") from error


def _file_sha256(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomically(destination: Path, data: dict[str, Any]) -> None:
    temporary_path = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(destination)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise


_SCHEMA = """
CREATE TABLE courses (
    id TEXT PRIMARY KEY,
    course_name TEXT NOT NULL,
    course_code TEXT NOT NULL COLLATE NOCASE UNIQUE,
    syllabus_pdf TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    archived_at TEXT
);

CREATE TABLE syllabus_versions (
    id TEXT PRIMARY KEY,
    course_id TEXT REFERENCES courses(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL CHECK (version_number > 0),
    course_name TEXT,
    course_code TEXT COLLATE NOCASE,
    source_pdf_path TEXT NOT NULL,
    source_sha256 TEXT,
    parsed_json_path TEXT,
    content_json TEXT NOT NULL,
    review_status TEXT NOT NULL CHECK (review_status IN ('pending', 'reviewed')),
    imported_at TEXT NOT NULL,
    reviewed_at TEXT,
    UNIQUE (course_id, course_code, version_number)
);

CREATE INDEX idx_syllabus_course_imported
    ON syllabus_versions(course_id, imported_at DESC);
CREATE INDEX idx_syllabus_code_imported
    ON syllabus_versions(course_code, imported_at DESC);

CREATE TABLE course_outcomes (
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    outcome_id TEXT NOT NULL CHECK (
        outcome_id IN ('CO1', 'CO2', 'CO3', 'CO4', 'CO5')
    ),
    description TEXT NOT NULL CHECK (length(trim(description)) > 0),
    source_pages_json TEXT NOT NULL DEFAULT '[]',
    source_syllabus_version_id TEXT REFERENCES syllabus_versions(id) ON DELETE SET NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (course_id, outcome_id)
);

CREATE TABLE questions (
    id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL CHECK (length(trim(question_text)) > 0),
    marks REAL CHECK (marks IS NULL OR marks >= 0),
    question_type TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX idx_questions_course_created
    ON questions(course_id, created_at DESC);

CREATE TABLE predictions (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    syllabus_version_id TEXT REFERENCES syllabus_versions(id) ON DELETE SET NULL,
    co_label TEXT NOT NULL CHECK (
        co_label IN ('CO1', 'CO2', 'CO3', 'CO4', 'CO5', 'N/A')
    ),
    bl_label TEXT NOT NULL CHECK (bl_label IN ('BL0', 'BL1', 'BL2', 'BL3', 'BL4', 'BL5', 'N/A')),
    status TEXT NOT NULL CHECK (
        status IN ('predicted', 'insufficient_context', 'failed')
    ),
    provider_name TEXT,
    model_name TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (id, question_id)
);
CREATE INDEX idx_predictions_question_created
    ON predictions(question_id, created_at DESC);

CREATE TABLE approved_labels (
    id TEXT PRIMARY KEY,
    question_id TEXT NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    prediction_id TEXT,
    co_label TEXT NOT NULL CHECK (
        co_label IN ('CO1', 'CO2', 'CO3', 'CO4', 'CO5', 'N/A')
    ),
    bl_label TEXT NOT NULL CHECK (bl_label IN ('BL0', 'BL1', 'BL2', 'BL3', 'BL4', 'BL5', 'N/A')),
    review_status TEXT NOT NULL CHECK (review_status IN ('approved', 'corrected')),
    reviewed_by TEXT,
    review_note TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (prediction_id, question_id)
        REFERENCES predictions(id, question_id) ON DELETE CASCADE
);
CREATE INDEX idx_approved_question_created
    ON approved_labels(question_id, created_at DESC);
"""
