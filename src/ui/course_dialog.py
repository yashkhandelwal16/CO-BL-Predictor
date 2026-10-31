from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.courses.course_manager import COURSE_OUTCOME_IDS, Course
from src.syllabus.pdf_extractor import PDFExtractionError, extract_syllabus
from src.ui.syllabus_review_dialog import SyllabusReviewDialog


class CourseDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, course: Course | None = None):
        super().__init__(parent)
        self.setWindowTitle("Edit course" if course else "Create course")
        self.setMinimumWidth(560)
        self.setStyleSheet(
            """
            QFrame#syllabusPicker {
                background-color: #faf9ff;
                border: 1px solid #7C3AED;
                border-radius: 9px;
            }
            QLabel#syllabusFileLabel {
                background: transparent;
                border: none;
                color: #625c6e;
                padding: 2px 4px;
            }
            QPushButton#chooseSyllabusPdfButton {
                background: #ffffff;
                border: 1px solid #7C3AED;
                border-radius: 7px;
                color: #6d28d9;
                font-weight: 600;
                padding: 0 16px;
            }
            QPushButton#chooseSyllabusPdfButton:hover {
                background: #f5f3ff;
            }
            QPushButton#chooseSyllabusPdfButton:pressed {
                background: #ede9fe;
            }
            """
        )
        self._syllabus_pdf = course.syllabus_pdf if course else None
        self._reviewed_syllabus: dict[str, object] | None = None
        self._outcome_inputs: dict[str, QTextEdit] = {}

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.name_input = QLineEdit(course.name if course else "")
        self.name_input.setPlaceholderText("For example, Data Structures")
        self.code_input = QLineEdit(course.code if course else "")
        self.code_input.setPlaceholderText("For example, CS402")
        form.addRow("Course name *", self.name_input)
        form.addRow("Course code *", self.code_input)

        for outcome_id in COURSE_OUTCOME_IDS:
            outcome_input = QTextEdit()
            outcome_input.setPlaceholderText(
                f"Enter the official {outcome_id} description"
            )
            outcome_input.setMinimumHeight(64)
            if course:
                outcome_input.setPlainText(
                    course.course_outcomes.get(outcome_id, "")
                )
            self._outcome_inputs[outcome_id] = outcome_input
            form.addRow(outcome_id, outcome_input)

        layout.addLayout(form)

        syllabus_heading = QLabel("Syllabus PDF (optional)")
        syllabus_heading.setObjectName("fieldLabel")
        layout.addWidget(syllabus_heading)

        self.syllabus_picker = QFrame()
        self.syllabus_picker.setObjectName("syllabusPicker")
        self.syllabus_picker.setFrameShape(QFrame.Shape.NoFrame)
        syllabus_picker_layout = QHBoxLayout(self.syllabus_picker)
        syllabus_picker_layout.setContentsMargins(14, 12, 14, 12)
        syllabus_picker_layout.setSpacing(16)
        self.syllabus_label = QLabel(
            Path(self._syllabus_pdf).name if self._syllabus_pdf else "No PDF selected"
        )
        self.syllabus_label.setObjectName("syllabusFileLabel")
        self.syllabus_label.setWordWrap(True)
        self.syllabus_label.setMinimumHeight(40)
        self.syllabus_label.setAlignment(
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft
        )
        self.syllabus_review_notice = QLabel("")
        self.syllabus_review_notice.setObjectName("noticeText")
        self.syllabus_review_notice.setWordWrap(True)
        self.syllabus_review_notice.hide()
        self.choose_pdf_button = QPushButton("Choose syllabus PDF")
        self.choose_pdf_button.setObjectName("chooseSyllabusPdfButton")
        self.choose_pdf_button.setMinimumHeight(42)
        self.choose_pdf_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.choose_pdf_button.setToolTip(
            "Select an optional PDF syllabus for this course"
        )
        self.choose_pdf_button.clicked.connect(self._choose_pdf)
        syllabus_picker_layout.addWidget(self.syllabus_label, 1)
        syllabus_picker_layout.addWidget(self.choose_pdf_button)
        layout.addWidget(self.syllabus_picker)
        layout.addWidget(self.syllabus_review_notice)

        help_text = QLabel(
            "Enter any available CO1–CO5 descriptions. Importing a syllabus "
            "opens a full review of its course details, units, and outcomes."
        )
        help_text.setObjectName("mutedText")
        help_text.setWordWrap(True)
        layout.addSpacing(4)
        layout.addWidget(help_text)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def course_values(self) -> tuple[str, str, dict[str, str], str | None]:
        outcomes = {
            outcome_id: field.toPlainText()
            for outcome_id, field in self._outcome_inputs.items()
        }
        return (
            self.name_input.text(),
            self.code_input.text(),
            outcomes,
            self._syllabus_pdf,
        )

    def reviewed_syllabus_data(self) -> dict[str, object] | None:
        if self._reviewed_syllabus is None:
            return None

        data = dict(self._reviewed_syllabus)
        original_outcomes = self._reviewed_syllabus.get("course_outcomes", [])
        if not isinstance(original_outcomes, list):
            original_outcomes = []
        original_by_id = {
            str(outcome.get("id", "")).strip().upper(): dict(outcome)
            for outcome in original_outcomes
            if isinstance(outcome, dict)
        }

        outcomes = []
        for outcome_id, original in original_by_id.items():
            if outcome_id in COURSE_OUTCOME_IDS:
                description = self._outcome_inputs[outcome_id].toPlainText().strip()
                if not description:
                    continue
                original["description"] = description
            outcomes.append(original)

        for outcome_id in COURSE_OUTCOME_IDS:
            if outcome_id in original_by_id:
                continue
            description = self._outcome_inputs[outcome_id].toPlainText().strip()
            if description:
                outcomes.append(
                    {
                        "id": outcome_id,
                        "description": description,
                        "source_pages": [],
                    }
                )

        data["course_name"] = self.name_input.text().strip() or None
        data["course_code"] = self.code_input.text().strip() or None
        data["course_outcomes"] = outcomes
        return data

    def _choose_pdf(self) -> None:
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select this course's syllabus PDF",
            "",
            "PDF files (*.pdf)",
        )
        if not selected_path:
            return

        try:
            syllabus_data = extract_syllabus(selected_path)
        except PDFExtractionError as error:
            QMessageBox.warning(self, "Syllabus could not be read", str(error))
            return

        review_dialog = SyllabusReviewDialog(syllabus_data, self)
        if review_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._reviewed_syllabus = review_dialog.syllabus_data()
        self._syllabus_pdf = selected_path
        self.syllabus_label.setText(Path(selected_path).name)
        if self._reviewed_syllabus["course_name"]:
            self.name_input.setText(str(self._reviewed_syllabus["course_name"]))
        if self._reviewed_syllabus["course_code"]:
            self.code_input.setText(str(self._reviewed_syllabus["course_code"]))
        extracted_outcomes = {
            outcome["id"]: outcome["description"]
            for outcome in self._reviewed_syllabus["course_outcomes"]
        }
        for outcome_id, description in extracted_outcomes.items():
            field = self._outcome_inputs.get(outcome_id)
            if field is not None:
                field.setPlainText(description)

        warnings = self._reviewed_syllabus.get("warnings", [])
        if warnings:
            self.syllabus_review_notice.setText(
                "Please review the extracted details. "
                + " ".join(str(warning) for warning in warnings)
            )
        elif extracted_outcomes:
            self.syllabus_review_notice.setText(
                "Course outcomes were filled from the document. Please verify them."
            )
        else:
            self.syllabus_review_notice.setText(
                "No course outcomes were found. Enter them manually."
            )
        self.syllabus_review_notice.show()
