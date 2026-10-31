from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QVBoxLayout,
)

from src.ui.selectable_text import enable_text_selection


class LabelReviewDialog(QDialog):
    def __init__(
        self,
        question_text: str,
        predicted_co: str | None,
        predicted_bl: str | None,
        approved_co: str | None = None,
        approved_bl: str | None = None,
        available_co_labels: tuple[str, ...] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review suggested labels")
        self.setMinimumWidth(440)

        layout = QVBoxLayout(self)
        question = QLabel(question_text)
        question.setWordWrap(True)
        layout.addWidget(question)

        guidance = QLabel(
            "The suggestion is kept in history. Your reviewed labels are saved "
            "separately and may be used as examples for this course."
        )
        guidance.setWordWrap(True)
        layout.addWidget(guidance)

        form = QFormLayout()
        self.co_selector = QComboBox()
        self.co_selector.addItems(
            [
                *(
                    available_co_labels
                    if available_co_labels is not None
                    else ("CO1", "CO2", "CO3", "CO4", "CO5")
                ),
                "N/A",
            ]
        )
        self.bl_selector = QComboBox()
        self.bl_selector.addItems(["BL0", "BL1", "BL2", "BL3", "BL4", "N/A"])
        self.reviewer_input = QLineEdit()
        self.reviewer_input.setPlaceholderText("Optional")
        self.review_note_input = QTextEdit()
        self.review_note_input.setPlaceholderText("Optional")
        self.review_note_input.setMaximumHeight(90)
        form.addRow("Course outcome", self.co_selector)
        form.addRow("Cognitive level", self.bl_selector)
        form.addRow("Reviewed by", self.reviewer_input)
        form.addRow("Review note", self.review_note_input)
        layout.addLayout(form)

        self._select_value(
            self.co_selector, approved_co or predicted_co or "N/A"
        )
        self._select_value(
            self.bl_selector, approved_bl or predicted_bl or "N/A"
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Save
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        enable_text_selection(self)

    @staticmethod
    def _select_value(selector: QComboBox, value: str) -> None:
        index = selector.findText(value)
        if index >= 0:
            selector.setCurrentIndex(index)

    @property
    def co_label(self) -> str:
        return self.co_selector.currentText()

    @property
    def bl_label(self) -> str:
        return self.bl_selector.currentText()

    @property
    def reviewed_by(self) -> str:
        return self.reviewer_input.text().strip()

    @property
    def review_note(self) -> str:
        return self.review_note_input.toPlainText().strip()
