import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pymupdf
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
)

from src.courses.course_manager import COURSE_OUTCOME_IDS
from src.syllabus.pdf_extractor import validate_syllabus_data
from src.ui.course_dialog import CourseDialog
from src.ui.main_window import MainWindow
from src.ui.syllabus_review_dialog import SyllabusReviewDialog
import src.ui.main_window as main_window_module


def get_application() -> QApplication:
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    return application


def test_window_has_all_phase_one_pages() -> None:
    get_application()
    window = MainWindow()

    assert set(window.pages) == {
        "Dashboard",
        "Courses",
        "Syllabus",
        "Predict",
        "History",
    }
    assert window.minimumWidth() >= 900
    assert window.minimumHeight() >= 600


def test_navigation_changes_page_and_header() -> None:
    get_application()
    window = MainWindow()

    window.navigation_buttons["Predict"].click()

    assert window.page_stack.currentWidget() is window.pages["Predict"]
    assert window.page_title.text() == "Predict"
    assert window.navigation_buttons["Predict"].isChecked()


def test_prediction_requires_a_course_and_question() -> None:
    get_application()
    window = MainWindow()

    assert not window.predict_button.isEnabled()
    assert window.course_selector.count() == 0

    outcomes = {
        outcome_id: f"Description for {outcome_id}"
        for outcome_id in COURSE_OUTCOME_IDS
    }
    window.course_manager.add_course("Computer Networks", "CS305", outcomes)
    window.refresh_course_list()
    assert window.course_selector.count() == 1
    assert window.course_selector.currentData(Qt.ItemDataRole.UserRole) == "CS305"
    assert not window.predict_button.isEnabled()

    window.question_input.setPlainText("Explain network routing.")
    assert window.predict_button.isEnabled()


def test_prediction_result_displays_only_the_required_two_lines() -> None:
    get_application()
    window = MainWindow()

    window._prediction_completed("CO level-CO2\nBL level - BL3")

    assert window.prediction_result.text() == "CO level-CO2\nBL level - BL3"
    assert window.prediction_status.text() == "Suggestion ready for faculty review."


def test_end_user_interface_hides_implementation_details() -> None:
    get_application()
    window = MainWindow()
    dialog = CourseDialog(window)

    widgets = (
        window.findChildren(QLabel)
        + window.findChildren(QPushButton)
        + dialog.findChildren(QLabel)
        + dialog.findChildren(QPushButton)
        + dialog.findChildren(QLineEdit)
        + dialog.findChildren(QTextEdit)
    )
    visible_copy = []
    for widget in widgets:
        if isinstance(widget, (QLabel, QPushButton, QLineEdit)):
            visible_copy.append(widget.text())
        else:
            visible_copy.append(widget.placeholderText())
    visible_copy.append(window.statusBar().currentMessage())
    visible_copy = " ".join(visible_copy).casefold()

    for internal_term in (
        "prototype",
        "phase 1",
        "backend",
        "database",
        "model",
        "sqlite",
        "llama",
        "qwen",
        "in-memory",
    ):
        assert internal_term not in visible_copy

    assert not window.navigation_buttons.get("Settings")


def test_course_syllabus_picker_has_clear_bordered_upload_area() -> None:
    get_application()
    dialog = CourseDialog()

    assert dialog.syllabus_picker.objectName() == "syllabusPicker"
    assert dialog.syllabus_label.text() == "No PDF selected"
    assert dialog.choose_pdf_button.text() == "Choose syllabus PDF"
    assert dialog.choose_pdf_button.minimumHeight() >= 40
    assert dialog.syllabus_review_notice.isHidden()
    assert dialog.syllabus_picker.frameShape() == QFrame.Shape.NoFrame
    assert "border: 1px solid #7C3AED" in dialog.styleSheet()
    assert dialog.syllabus_picker.layout().spacing() >= 12


def test_main_window_uses_white_and_violet_theme() -> None:
    get_application()
    window = MainWindow()
    stylesheet = window.styleSheet()

    assert "background: #ffffff" in stylesheet
    assert "#7C3AED" in stylesheet
    assert "background: #f5f3ff" in stylesheet
    assert "QComboBox, QLineEdit, QTextEdit" in stylesheet
    assert "QLabel#brand" in stylesheet
    assert "color: #000000" in stylesheet
    assert "QPushButton#navigationButton:checked" in stylesheet
    assert "color: #ffffff" in stylesheet

    active_navigation = window.navigation_buttons["Dashboard"]
    assert active_navigation.isChecked()


def test_course_page_lists_and_filters_courses() -> None:
    get_application()
    window = MainWindow()
    outcomes = {
        outcome_id: f"Description for {outcome_id}"
        for outcome_id in COURSE_OUTCOME_IDS
    }
    window.course_manager.add_course("Data Structures", "CS301", outcomes)
    window.course_manager.add_course("Database Systems", "CS402", outcomes)

    window.refresh_course_list()
    assert window.course_list.count() == 2

    window.course_search.setText("cs402")
    assert window.course_list.count() == 1
    assert (
        window.course_list.currentItem().data(Qt.ItemDataRole.UserRole) == "CS402"
    )
    assert window.course_detail_name.text() == "Database Systems"


def test_persistent_window_restores_courses_syllabi_and_question_history(
    tmp_path,
) -> None:
    get_application()
    database_path = tmp_path / "predictor.sqlite3"
    first_window = MainWindow(database_path)
    outcomes = {
        outcome_id: f"Description for {outcome_id}"
        for outcome_id in COURSE_OUTCOME_IDS
    }
    first_window.course_manager.add_course(
        "Computer Networks", "CS305", outcomes
    )
    syllabus_path = tmp_path / "networks.pdf"
    first_window.database.save_syllabus_version(
        {
            "course_name": "Computer Networks",
            "course_code": "CS305",
            "source_file": str(syllabus_path),
            "course_outcomes": [
                {"id": outcome_id, "description": outcomes[outcome_id]}
                for outcome_id in COURSE_OUTCOME_IDS
            ],
            "units": [],
            "review_status": "reviewed",
        },
        tmp_path / "networks.json",
    )
    question = first_window.database.create_question(
        "CS305", "Explain packet routing."
    )
    prediction = first_window.database.create_prediction(
        question["id"], "CO2", "BL2"
    )
    first_window.database.save_approved_labels(
        question["id"],
        "CO3",
        "BL3",
        prediction_id=prediction["id"],
        review_status="corrected",
    )
    first_window.close_storage()

    restarted_window = MainWindow(database_path)

    assert restarted_window.course_list.count() == 1
    restored_course = restarted_window.course_manager.get_course("CS305")
    assert restored_course is not None
    assert restored_course.course_outcomes == outcomes
    assert restarted_window.syllabus_versions_list.count() == 1
    assert restarted_window.history_table.rowCount() == 1
    assert restarted_window.history_table.item(0, 3).text() == "CO3 / BL3"
    assert restarted_window.history_table.item(0, 4).text() == "corrected"
    restarted_window.close_storage()


def test_faculty_review_saves_correction_without_changing_prediction(
    monkeypatch,
) -> None:
    get_application()
    window = MainWindow()
    outcomes = {
        outcome_id: f"Description for {outcome_id}"
        for outcome_id in COURSE_OUTCOME_IDS
    }
    window.course_manager.add_course("Computer Networks", "CS305", outcomes)
    question = window.database.create_question(
        "CS305", "Compare two routing algorithms."
    )
    prediction = window.database.create_prediction(
        question["id"], "CO2", "BL2"
    )
    window.refresh_question_history()

    assert not window.review_question_button.isEnabled()
    window.history_table.selectRow(0)
    assert window.review_question_button.isEnabled()

    def accept_correction(dialog):
        dialog.co_selector.setCurrentText("CO3")
        dialog.bl_selector.setCurrentText("BL3")
        dialog.reviewer_input.setText("Faculty reviewer")
        dialog.review_note_input.setPlainText("Reviewed against the syllabus.")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        main_window_module.LabelReviewDialog, "exec", accept_correction
    )
    window.review_question_button.click()

    history = window.database.list_question_history()
    assert history[0]["predicted_co"] == "CO2"
    assert history[0]["predicted_bl"] == "BL2"
    assert history[0]["approved_co"] == "CO3"
    assert history[0]["approved_bl"] == "BL3"
    assert history[0]["label_review_status"] == "corrected"
    with window.database._lock:
        saved = window.database._connection.execute(
            "SELECT prediction_id, review_status, reviewed_by, review_note "
            "FROM approved_labels WHERE question_id = ?",
            (question["id"],),
        ).fetchone()
    assert tuple(saved) == (
        prediction["id"],
        "corrected",
        "Faculty reviewer",
        "Reviewed against the syllabus.",
    )
    window.close_storage()


def test_course_delete_requires_confirmation(monkeypatch) -> None:
    get_application()
    window = MainWindow()
    outcomes = {
        outcome_id: f"Description for {outcome_id}"
        for outcome_id in COURSE_OUTCOME_IDS
    }
    window.course_manager.add_course("Database Systems", "CS402", outcomes)
    window.refresh_course_list()

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args: QMessageBox.StandardButton.No,
    )
    window._delete_selected_course()
    assert window.course_manager.get_course("CS402") is not None

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args: QMessageBox.StandardButton.Yes,
    )
    window._delete_selected_course()
    assert window.course_manager.get_course("CS402") is None


def test_course_dialog_keeps_four_outcomes_separate() -> None:
    get_application()
    dialog = CourseDialog()
    dialog.name_input.setText("Computer Networks")
    dialog.code_input.setText("CS305")
    for outcome_id, field in dialog._outcome_inputs.items():
        field.setPlainText(f"Description for {outcome_id}")

    name, code, outcomes, syllabus_pdf = dialog.course_values()

    assert (name, code) == ("Computer Networks", "CS305")
    assert outcomes == {
        outcome_id: f"Description for {outcome_id}"
        for outcome_id in COURSE_OUTCOME_IDS
    }
    assert syllabus_pdf is None


def test_create_course_action_connects_form_to_course_manager(monkeypatch) -> None:
    get_application()
    window = MainWindow()
    values = (
        "Computer Networks",
        "CS305",
        {
            outcome_id: f"Description for {outcome_id}"
            for outcome_id in COURSE_OUTCOME_IDS
        },
        None,
    )
    monkeypatch.setattr(
        CourseDialog,
        "exec",
        lambda self: QDialog.DialogCode.Accepted,
    )
    monkeypatch.setattr(CourseDialog, "course_values", lambda self: values)

    window._create_course()

    course = window.course_manager.get_course("CS305")
    assert course is not None
    assert course.name == "Computer Networks"
    assert window.course_list.count() == 1


def test_syllabus_review_dialog_allows_editing_and_validates() -> None:
    get_application()
    data = {
        "course_name": "Computer Networks",
        "course_code": "CS305",
        "units": [
            {
                "unit_number": 1,
                "unit_title": "Introduction",
                "topics": ["Network models"],
                "source_pages": [1],
            }
        ],
        "course_outcomes": [
            {
                "id": "CO1",
                "description": "Explain network models.",
                "source_pages": [1],
            }
        ],
        "pages": [{"page": 1, "text": "Source syllabus text"}],
        "source_file": r"C:\syllabi\networks.pdf",
        "review_status": "pending",
        "warnings": [
            "Course or subject name was not clearly identified.",
            "Course code was not clearly identified.",
            "No clear unit headings were found; review the PDF manually.",
            "Check all extracted details against the original syllabus before use.",
        ],
    }
    dialog = SyllabusReviewDialog(data)
    dialog.course_name_input.setText("Computer Networks (Revised)")
    dialog.units_table.item(0, 2).setText("Network models\nProtocol layers")
    dialog.outcomes_table.item(0, 1).setText("Describe network models.")

    reviewed = dialog.syllabus_data()

    assert reviewed["course_name"] == "Computer Networks (Revised)"
    assert reviewed["units"][0]["topics"] == [
        "Network models",
        "Protocol layers",
    ]
    assert reviewed["course_outcomes"][0]["description"] == (
        "Describe network models."
    )
    assert reviewed["review_status"] == "reviewed"
    assert reviewed["pages"] == data["pages"]
    assert reviewed["warnings"] == [
        "Check all extracted details against the original syllabus before use."
    ]
    validate_syllabus_data(reviewed)


def test_course_pdf_selection_populates_extracted_outcome_fields(
    tmp_path, monkeypatch
) -> None:
    get_application()
    pdf_path = tmp_path / "course.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Course Name: Networks\nCourse Code: CS301\n"
        "CO1: Explain network models.\nCO2: Analyze routing protocols.",
    )
    document.save(pdf_path)
    document.close()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args: (str(pdf_path), "PDF files (*.pdf)"),
    )
    dialog = CourseDialog()

    dialog._choose_pdf()

    assert dialog.syllabus_label.text() == "course.pdf"
    assert dialog._outcome_inputs["CO1"].toPlainText() == (
        "Explain network models."
    )
    assert dialog._outcome_inputs["CO2"].toPlainText() == (
        "Analyze routing protocols."
    )
    assert dialog.syllabus_review_notice.text()


def test_syllabus_import_reviews_and_saves_structured_data(
    tmp_path, monkeypatch
) -> None:
    get_application()
    pdf_path = tmp_path / "course.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Course Name: Networks\nCourse Code: CS301\n"
        "UNIT I: Basics\nRouting\nCO1: Explain network routing.",
    )
    document.save(pdf_path)
    document.close()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args: (str(pdf_path), "PDF files (*.pdf)"),
    )
    monkeypatch.setattr(
        SyllabusReviewDialog,
        "exec",
        lambda self: QDialog.DialogCode.Accepted,
    )
    saved_records = []
    saved_path = tmp_path / "reviewed.json"

    def save_review(data, output_directory=None):
        saved_records.append(data)
        return saved_path

    monkeypatch.setattr(main_window_module, "save_syllabus_json", save_review)
    window = MainWindow()

    window._import_syllabus()

    assert len(saved_records) == 1
    assert saved_records[0]["review_status"] == "reviewed"
    assert saved_records[0]["course_name"] == "Networks"
    assert saved_records[0]["units"][0]["topics"] == ["Routing"]
    assert window.syllabus_file_label.text() == "Reviewed: course.pdf"


def test_course_pdf_selection_warns_about_outcomes_outside_four_course_fields(
    tmp_path, monkeypatch
) -> None:
    get_application()
    pdf_path = tmp_path / "course.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Course Name: Networks\nCourse Code: CS301\n"
        "CO1: Explain routes.\nCO5: Evaluate networks.",
    )
    document.save(pdf_path)
    document.close()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args: (str(pdf_path), "PDF files (*.pdf)"),
    )
    dialog = CourseDialog()

    dialog._choose_pdf()

    assert "CO5" in dialog.syllabus_review_notice.text()
    assert dialog._outcome_inputs["CO1"].toPlainText() == "Explain routes."
