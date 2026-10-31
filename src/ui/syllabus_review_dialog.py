from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.syllabus.pdf_extractor import validate_syllabus_data
from src.ui.selectable_text import enable_text_selection


class SyllabusReviewDialog(QDialog):
    def __init__(
        self,
        syllabus_data: dict[str, Any],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review syllabus details")
        self.setMinimumSize(820, 620)
        self._source_file = syllabus_data["source_file"]
        self._original_pages = syllabus_data["pages"]
        self._warnings = list(syllabus_data["warnings"])
        supported_outcomes = {f"CO{number}" for number in range(1, 6)}
        unsupported_outcomes = sorted(
            {
                str(outcome.get("id", "")).strip().upper()
                for outcome in syllabus_data["course_outcomes"]
                if isinstance(outcome, dict)
            }.difference(supported_outcomes)
        )
        if unsupported_outcomes:
            self._warnings.append(
                "Outcomes outside CO1–CO5 remain in the syllabus archive but "
                "cannot be used for prediction: "
                + ", ".join(unsupported_outcomes)
                + "."
            )

        layout = QVBoxLayout(self)
        source_label = QLabel(
            f"Original document: {Path(self._source_file).name}"
        )
        source_label.setObjectName("mutedText")
        source_label.setWordWrap(True)
        layout.addWidget(source_label)

        if self._warnings:
            warnings_label = QLabel(
                "Please review: " + " ".join(self._warnings)
            )
            warnings_label.setObjectName("noticeText")
            warnings_label.setWordWrap(True)
            layout.addWidget(warnings_label)

        tabs = QTabWidget()
        tabs.addTab(self._course_details_tab(syllabus_data), "Course details")
        tabs.addTab(self._units_tab(syllabus_data), "Units and topics")
        tabs.addTab(self._outcomes_tab(syllabus_data), "Course outcomes")
        tabs.addTab(self._source_text_tab(syllabus_data), "Source text")
        layout.addWidget(tabs, 1)

        buttons = QDialogButtonBox()
        save_button = buttons.addButton(
            "Save Course",
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        cancel_button = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        save_button.clicked.connect(self._validate_and_accept)
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(buttons)
        enable_text_selection(self)

    def _course_details_tab(self, data: dict[str, Any]) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.course_name_input = QLineEdit(data.get("course_name") or "")
        self.course_code_input = QLineEdit(data.get("course_code") or "")
        form.addRow("Course or subject name", self.course_name_input)
        form.addRow("Course code", self.course_code_input)
        return page

    def _units_tab(self, data: dict[str, Any]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.units_table = QTableWidget()
        self.units_table.setColumnCount(4)
        self.units_table.setHorizontalHeaderLabels(
            ["Unit", "Title", "Topics (one per line)", "Source pages"]
        )
        self.units_table.setRowCount(len(data["units"]))
        for row, unit in enumerate(data["units"]):
            values = (
                str(unit["unit_number"]),
                unit["unit_title"],
                "\n".join(unit["topics"]),
                ", ".join(str(page) for page in unit["source_pages"]),
            )
            for column, value in enumerate(values):
                self.units_table.setItem(row, column, QTableWidgetItem(value))
        self.units_table.horizontalHeader().setStretchLastSection(True)
        self.units_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )

        buttons = QHBoxLayout()
        add_button = QPushButton("Add unit")
        remove_button = QPushButton("Remove selected unit")
        add_button.clicked.connect(self._add_unit_row)
        remove_button.clicked.connect(self._remove_unit_rows)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        layout.addWidget(self.units_table)
        layout.addLayout(buttons)
        return page

    def _outcomes_tab(self, data: dict[str, Any]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.outcomes_table = QTableWidget()
        self.outcomes_table.setColumnCount(3)
        self.outcomes_table.setHorizontalHeaderLabels(
            ["Outcome ID", "Description", "Source pages"]
        )
        self.outcomes_table.setRowCount(len(data["course_outcomes"]))
        for row, outcome in enumerate(data["course_outcomes"]):
            values = (
                outcome["id"],
                outcome["description"],
                ", ".join(str(page) for page in outcome["source_pages"]),
            )
            for column, value in enumerate(values):
                self.outcomes_table.setItem(row, column, QTableWidgetItem(value))
        self.outcomes_table.horizontalHeader().setStretchLastSection(True)
        self.outcomes_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )

        buttons = QHBoxLayout()
        add_button = QPushButton("Add outcome")
        remove_button = QPushButton("Remove selected outcome")
        add_button.clicked.connect(self._add_outcome_row)
        remove_button.clicked.connect(self._remove_outcome_rows)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        layout.addWidget(self.outcomes_table)
        layout.addLayout(buttons)
        return page

    @staticmethod
    def _source_text_tab(data: dict[str, Any]) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        source_text = QPlainTextEdit()
        source_text.setReadOnly(True)
        source_text.setPlainText(
            "\n\n".join(
                f"--- Page {item['page']} ---\n{item['text']}"
                for item in data["pages"]
            )
        )
        layout.addWidget(source_text)
        return page

    def syllabus_data(self) -> dict[str, Any]:
        units = []
        for row in range(self.units_table.rowCount()):
            unit_number = self._cell_text(self.units_table, row, 0)
            title = self._cell_text(self.units_table, row, 1)
            topics = [
                topic.strip()
                for topic in self._cell_text(self.units_table, row, 2).splitlines()
                if topic.strip()
            ]
            pages = self._parse_page_numbers(
                self._cell_text(self.units_table, row, 3)
            )
            units.append(
                {
                    "unit_number": int(unit_number) if unit_number.isdigit() else 0,
                    "unit_title": title,
                    "topics": topics,
                    "source_pages": pages,
                }
            )

        outcomes = []
        for row in range(self.outcomes_table.rowCount()):
            outcomes.append(
                {
                    "id": self._cell_text(
                        self.outcomes_table, row, 0
                    ).upper(),
                    "description": self._cell_text(self.outcomes_table, row, 1),
                    "source_pages": self._parse_page_numbers(
                        self._cell_text(self.outcomes_table, row, 2)
                    ),
                }
            )

        warnings = list(self._warnings)
        if self.course_name_input.text().strip():
            warnings = [
                warning
                for warning in warnings
                if warning != "Course or subject name was not clearly identified."
            ]
        if self.course_code_input.text().strip():
            warnings = [
                warning
                for warning in warnings
                if warning != "Course code was not clearly identified."
            ]
        if units:
            warnings = [
                warning
                for warning in warnings
                if not warning.startswith("No clear unit headings were found")
            ]
        if outcomes:
            warnings = [
                warning
                for warning in warnings
                if not warning.startswith("No clearly labelled course outcomes")
            ]

        return {
            "course_name": self.course_name_input.text().strip() or None,
            "course_code": self.course_code_input.text().strip() or None,
            "units": units,
            "course_outcomes": outcomes,
            "pages": self._original_pages,
            "source_file": self._source_file,
            "review_status": "reviewed",
            "warnings": warnings,
        }

    def _validate_and_accept(self) -> None:
        if not self.course_name_input.text().strip():
            QMessageBox.warning(
                self, "Check course details", "Course name is required."
            )
            return
        if not self.course_code_input.text().strip():
            QMessageBox.warning(
                self, "Check course details", "Course code is required."
            )
            return
        try:
            validate_syllabus_data(self.syllabus_data())
        except (TypeError, ValueError) as error:
            QMessageBox.warning(self, "Check syllabus details", str(error))
            return
        self.accept()

    def _add_unit_row(self) -> None:
        row = self.units_table.rowCount()
        self.units_table.insertRow(row)
        self.units_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.units_table.setItem(row, 1, QTableWidgetItem(""))
        self.units_table.setItem(row, 2, QTableWidgetItem(""))
        self.units_table.setItem(row, 3, QTableWidgetItem(""))

    def _remove_unit_rows(self) -> None:
        self._remove_selected_rows(self.units_table)

    def _add_outcome_row(self) -> None:
        row = self.outcomes_table.rowCount()
        self.outcomes_table.insertRow(row)
        existing_ids = {
            self._cell_text(self.outcomes_table, index, 0).upper()
            for index in range(row)
        }
        next_outcome = next(
            (
                f"CO{number}"
                for number in range(1, 6)
                if f"CO{number}" not in existing_ids
            ),
            f"CO{row + 1}",
        )
        self.outcomes_table.setItem(row, 0, QTableWidgetItem(next_outcome))
        self.outcomes_table.setItem(row, 1, QTableWidgetItem(""))
        self.outcomes_table.setItem(row, 2, QTableWidgetItem(""))

    def _remove_outcome_rows(self) -> None:
        self._remove_selected_rows(self.outcomes_table)

    @staticmethod
    def _remove_selected_rows(table: QTableWidget) -> None:
        selected_rows = sorted(
            {index.row() for index in table.selectionModel().selectedRows()},
            reverse=True,
        )
        for row in selected_rows:
            table.removeRow(row)

    @staticmethod
    def _cell_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)
        return item.text().strip() if item is not None else ""

    @staticmethod
    def _parse_page_numbers(value: str) -> list[int]:
        if not value.strip():
            return []
        pages = []
        for item in value.split(","):
            normalized = item.strip()
            if not normalized.isdigit():
                raise ValueError("Source pages must be comma-separated numbers.")
            pages.append(int(normalized))
        return pages
