from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QComboBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.courses.course_manager import (
    COURSE_OUTCOME_IDS,
    Course,
    CourseManager,
    CourseValidationError,
)
from src.database.db_manager import DatabaseError, DatabaseManager
from src.prediction.bloom_rules import load_bloom_config
from src.prediction.prediction_worker import PredictionWorker
from src.syllabus.pdf_extractor import (
    PDFExtractionError,
    extract_syllabus,
    save_syllabus_json,
)
from src.ui.course_dialog import CourseDialog
from src.ui.label_review_dialog import LabelReviewDialog
from src.ui.syllabus_review_dialog import SyllabusReviewDialog


class MainWindow(QMainWindow):
    PAGES = (
        ("Dashboard", "Dashboard"),
        ("Courses", "Courses"),
        ("Syllabus", "Syllabus"),
        ("Predict", "Predict"),
        ("History", "History"),
    )

    def __init__(self, database_path: str | Path = ":memory:") -> None:
        super().__init__()
        self.setWindowTitle("CO & BL Predictor")
        self.setMinimumSize(980, 640)
        self.resize(1180, 760)
        self.setStyleSheet(self._stylesheet())

        self.navigation_buttons: dict[str, QPushButton] = {}
        self.pages: dict[str, QWidget] = {}
        self.database = DatabaseManager(database_path)
        self.course_manager = CourseManager(self.database)
        self.bloom_config = load_bloom_config()
        self._prediction_worker: PredictionWorker | None = None
        self.page_stack = QStackedWidget()

        central_widget = QWidget()
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_sidebar())

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(30, 24, 30, 22)
        content_layout.setSpacing(18)
        content_layout.addWidget(self._build_header())
        content_layout.addWidget(self.page_stack, 1)
        root_layout.addWidget(content, 1)
        self.setCentralWidget(central_widget)

        self._build_pages()
        self.refresh_prediction_courses()
        self.statusBar().showMessage("Welcome. Add a course to get started.")
        self.navigate("Dashboard")

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(224)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 24, 18, 20)
        layout.setSpacing(8)

        brand = QLabel("CO & BL\nPredictor")
        brand.setObjectName("brand")
        layout.addWidget(brand)

        caption = QLabel("ACADEMIC MAPPING WORKSPACE")
        caption.setObjectName("sidebarCaption")
        layout.addWidget(caption)
        layout.addSpacing(24)

        for label, page_name in self.PAGES:
            button = QPushButton(label)
            button.setObjectName("navigationButton")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(
                lambda checked=False, name=page_name: self.navigate(name)
            )
            self.navigation_buttons[page_name] = button
            layout.addWidget(button)

        layout.addStretch(1)

        return sidebar

    def _build_header(self) -> QWidget:
        header = QWidget()
        layout = QHBoxLayout(header)
        layout.setContentsMargins(0, 0, 0, 0)

        self.page_title = QLabel("Dashboard")
        self.page_title.setObjectName("pageTitle")
        layout.addWidget(self.page_title)
        layout.addStretch(1)
        return header

    def _build_pages(self) -> None:
        self.pages["Dashboard"] = self._dashboard_page()
        self.pages["Courses"] = self._courses_page()
        self.pages["Syllabus"] = self._syllabus_page()
        self.pages["Predict"] = self._predict_page()
        self.pages["History"] = self._history_page()

        for page in self.pages.values():
            self.page_stack.addWidget(page)

    def _dashboard_page(self) -> QWidget:
        page, layout = self._new_page(
            "Welcome",
            "Organize course outcomes and prepare questions for academic review.",
        )

        notice = self._card()
        notice_layout = QVBoxLayout(notice)
        notice_layout.setContentsMargins(22, 20, 22, 20)
        heading = QLabel("Start with a course")
        heading.setObjectName("cardTitle")
        body = QLabel(
            "Add a course and its official outcomes to get started. Suggested "
            "classifications are intended to support faculty review."
        )
        body.setObjectName("bodyText")
        body.setWordWrap(True)
        notice_layout.addWidget(heading)
        notice_layout.addWidget(body)
        layout.addWidget(notice)

        actions_card = self._card()
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(22, 18, 22, 18)
        action_title = QLabel("Quick actions")
        action_title.setObjectName("cardTitle")
        actions_layout.addWidget(action_title)
        buttons = QHBoxLayout()
        for text, destination in (
            ("Manage courses", "Courses"),
            ("Review syllabus", "Syllabus"),
            ("Open prediction", "Predict"),
        ):
            button = QPushButton(text)
            button.setObjectName("secondaryButton")
            button.clicked.connect(
                lambda checked=False, name=destination: self.navigate(name)
            )
            buttons.addWidget(button)
        actions_layout.addLayout(buttons)
        layout.addWidget(actions_card)
        layout.addStretch(1)
        return page

    def _courses_page(self) -> QWidget:
        page, layout = self._new_page(
            "Courses",
            "Manage courses and their official course outcome descriptions.",
        )

        toolbar = QHBoxLayout()
        self.course_search = QLineEdit()
        self.course_search.setPlaceholderText("Search by course name or code")
        self.course_search.textChanged.connect(self.refresh_course_list)
        toolbar.addWidget(self.course_search, 1)

        create_button = QPushButton("Create course")
        create_button.setObjectName("primaryButton")
        create_button.clicked.connect(self._create_course)
        toolbar.addWidget(create_button)
        layout.addLayout(toolbar)

        columns = QHBoxLayout()
        columns.setSpacing(14)

        list_card = self._card()
        list_layout = QVBoxLayout(list_card)
        list_layout.setContentsMargins(14, 14, 14, 14)
        list_heading = QLabel("Courses")
        list_heading.setObjectName("cardTitle")
        self.course_list = QListWidget()
        self.course_list.currentItemChanged.connect(self._display_selected_course)
        list_layout.addWidget(list_heading)
        self.course_empty_label = QLabel(
            "No courses yet. Select “Create course” to add one."
        )
        self.course_empty_label.setObjectName("mutedText")
        self.course_empty_label.setWordWrap(True)
        list_layout.addWidget(self.course_empty_label)
        list_layout.addWidget(self.course_list)

        detail_card = self._card()
        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(18, 18, 18, 18)
        self.course_detail_name = QLabel("Select a course")
        self.course_detail_name.setObjectName("cardTitle")
        self.course_detail_code = QLabel("")
        self.course_detail_code.setObjectName("mutedText")
        self.course_detail_syllabus = QLabel("")
        self.course_detail_syllabus.setObjectName("mutedText")
        self.course_detail_syllabus.setWordWrap(True)
        detail_layout.addWidget(self.course_detail_name)
        detail_layout.addWidget(self.course_detail_code)
        detail_layout.addWidget(self.course_detail_syllabus)
        detail_layout.addSpacing(8)

        self.course_outcome_labels: dict[str, QLabel] = {}
        for outcome_id in COURSE_OUTCOME_IDS:
            label = QLabel(f"{outcome_id}:")
            label.setObjectName("bodyText")
            label.setWordWrap(True)
            self.course_outcome_labels[outcome_id] = label
            detail_layout.addWidget(label)

        detail_layout.addStretch(1)
        course_actions = QHBoxLayout()
        self.edit_course_button = QPushButton("Edit")
        self.edit_course_button.setObjectName("secondaryButton")
        self.edit_course_button.clicked.connect(self._edit_selected_course)
        self.delete_course_button = QPushButton("Delete")
        self.delete_course_button.setObjectName("secondaryButton")
        self.delete_course_button.clicked.connect(self._delete_selected_course)
        self.edit_course_button.setEnabled(False)
        self.delete_course_button.setEnabled(False)
        course_actions.addWidget(self.edit_course_button)
        course_actions.addWidget(self.delete_course_button)
        detail_layout.addLayout(course_actions)

        columns.addWidget(list_card, 2)
        columns.addWidget(detail_card, 3)
        layout.addLayout(columns, 1)
        self.refresh_course_list()
        layout.addStretch(1)
        return page

    def refresh_course_list(
        self,
        _query: str | None = None,
        preferred_code: str | None = None,
    ) -> None:
        selected_code = preferred_code or self._selected_course_code()
        query = self.course_search.text() if hasattr(self, "course_search") else ""
        courses = self.course_manager.search(query)

        self.course_list.blockSignals(True)
        self.course_list.clear()
        selected_item = None
        for course in courses:
            item = QListWidgetItem(f"{course.name}\n{course.code}")
            item.setData(Qt.ItemDataRole.UserRole, course.code)
            self.course_list.addItem(item)
            if course.code == selected_code:
                selected_item = item
        if selected_item is not None:
            self.course_list.setCurrentItem(selected_item)
        elif self.course_list.count():
            self.course_list.setCurrentRow(0)
        self.course_list.blockSignals(False)
        self.course_empty_label.setVisible(not courses)

        self._display_selected_course(self.course_list.currentItem(), None)
        self.refresh_prediction_courses()

    def refresh_prediction_courses(self) -> None:
        if not hasattr(self, "course_selector"):
            return
        selected_code = self.course_selector.currentData(Qt.ItemDataRole.UserRole)
        courses = self.course_manager.list_courses()
        self.course_selector.blockSignals(True)
        self.course_selector.clear()
        for course in courses:
            self.course_selector.addItem(
                f"{course.name} ({course.code})", course.code
            )
        if selected_code:
            selected_index = self.course_selector.findData(
                selected_code, Qt.ItemDataRole.UserRole
            )
            if selected_index >= 0:
                self.course_selector.setCurrentIndex(selected_index)
        elif courses:
            self.course_selector.setCurrentIndex(0)
        self.course_selector.blockSignals(False)
        self.course_selector.setEnabled(bool(courses))
        self._update_prediction_button()

    def _update_prediction_button(self) -> None:
        if not hasattr(self, "predict_button"):
            return
        has_course = (
            self.course_selector.currentData(Qt.ItemDataRole.UserRole) is not None
        )
        has_question = bool(self.question_input.toPlainText().strip())
        self.predict_button.setEnabled(
            has_course and has_question and self._prediction_worker is None
        )

    def _selected_course_code(self) -> str | None:
        item = self.course_list.currentItem() if hasattr(self, "course_list") else None
        if item is None:
            return None
        return item.data(Qt.ItemDataRole.UserRole)

    def _display_selected_course(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        course = None
        if current is not None:
            course_code = current.data(Qt.ItemDataRole.UserRole)
            course = self.course_manager.get_course(course_code)

        has_selection = course is not None
        self.edit_course_button.setEnabled(has_selection)
        self.delete_course_button.setEnabled(has_selection)
        if course is None:
            self.course_detail_name.setText("No course selected")
            self.course_detail_code.setText("")
            self.course_detail_syllabus.setText("")
            for outcome_id, label in self.course_outcome_labels.items():
                label.setText(f"{outcome_id}:")
            return

        self.course_detail_name.setText(course.name)
        self.course_detail_code.setText(f"Course code: {course.code}")
        syllabus = Path(course.syllabus_pdf).name if course.syllabus_pdf else "Not attached"
        versions = self.database.list_syllabus_versions(course.code)
        if versions:
            latest = versions[0]
            syllabus = (
                f"{Path(latest['source_pdf_path']).name} · "
                f"Version {latest['version_number']} · "
                f"{latest['review_status'].title()}"
            )
        self.course_detail_syllabus.setText(f"Syllabus PDF: {syllabus}")
        for outcome_id, label in self.course_outcome_labels.items():
            label.setText(f"{outcome_id}: {course.course_outcomes[outcome_id]}")

    def _create_course(self) -> None:
        self._edit_course()

    def _edit_selected_course(self) -> None:
        course_code = self._selected_course_code()
        course = self.course_manager.get_course(course_code) if course_code else None
        if course is not None:
            self._edit_course(course)

    def _edit_course(self, course: Course | None = None) -> None:
        dialog = CourseDialog(self, course)
        while dialog.exec() == QDialog.DialogCode.Accepted:
            name, code, outcomes, syllabus_pdf = dialog.course_values()
            try:
                if course is None:
                    saved_course = self.course_manager.add_course(
                        name, code, outcomes, syllabus_pdf
                    )
                else:
                    saved_course = self.course_manager.update_course(
                        course.code, name, code, outcomes, syllabus_pdf
                    )
            except CourseValidationError as error:
                QMessageBox.warning(self, "Course not saved", str(error))
                continue

            self.course_search.clear()
            self.refresh_course_list(preferred_code=saved_course.code)
            self.statusBar().showMessage(f"Saved course {saved_course.code}.", 4000)
            return

    def _delete_selected_course(self) -> None:
        course_code = self._selected_course_code()
        course = self.course_manager.get_course(course_code) if course_code else None
        if course is None:
            return

        answer = QMessageBox.question(
            self,
            "Delete course",
            f"Delete {course.name} ({course.code}), including its syllabus "
            "versions, questions, and review history?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        try:
            self.course_manager.delete_course(course.code)
        except CourseValidationError as error:
            QMessageBox.critical(self, "Course could not be deleted", str(error))
            return
        self.refresh_course_list()
        self.statusBar().showMessage(f"Deleted course {course.code}.", 4000)

    def _syllabus_page(self) -> QWidget:
        page, layout = self._new_page(
            "Syllabus",
            "Keep course documents organized for review.",
        )
        card = self._card()
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(22, 20, 22, 20)

        title = QLabel("Select an official syllabus PDF")
        title.setObjectName("cardTitle")
        description = QLabel(
            "Review course details, units, topics, and outcomes extracted from "
            "a syllabus. Check every detail against the original document."
        )
        description.setObjectName("bodyText")
        description.setWordWrap(True)
        self.syllabus_file_label = QLabel("No syllabus reviewed")
        self.syllabus_file_label.setObjectName("mutedText")
        self.syllabus_file_label.setWordWrap(True)
        self.syllabus_versions_list = QListWidget()
        self.syllabus_versions_list.setObjectName("syllabusVersions")

        choose_button = QPushButton("Import syllabus")
        choose_button.setObjectName("primaryButton")
        choose_button.clicked.connect(self._import_syllabus)

        card_layout.addWidget(title)
        card_layout.addWidget(description)
        card_layout.addWidget(self.syllabus_file_label)
        card_layout.addWidget(choose_button, 0, Qt.AlignmentFlag.AlignLeft)
        card_layout.addWidget(QLabel("Recently reviewed syllabi"))
        card_layout.addWidget(self.syllabus_versions_list)
        layout.addWidget(card)
        self.refresh_syllabus_versions()
        layout.addStretch(1)
        return page

    def _import_syllabus(self) -> None:
        selected_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an official syllabus PDF",
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

        dialog = SyllabusReviewDialog(syllabus_data, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        reviewed_data = dialog.syllabus_data()
        try:
            output_directory = (
                self.database.user_data_directory / "parsed_syllabi"
                if self.database.user_data_directory
                else None
            )
            saved_path = save_syllabus_json(reviewed_data, output_directory)
            version = self.database.save_syllabus_version(
                reviewed_data,
                saved_path,
            )
        except (DatabaseError, OSError, ValueError) as error:
            QMessageBox.critical(
                self,
                "Syllabus could not be saved",
                f"Check the syllabus details and available storage, then try again. "
                f"Your original PDF has not been changed.\n\n{error}",
            )
            return

        self.syllabus_file_label.setText(
            f"Reviewed: {Path(selected_path).name}"
        )
        self.refresh_syllabus_versions()
        self.refresh_course_list()
        self.statusBar().showMessage(
            f"Saved syllabus version {version['version_number']}.", 6000
        )

    def refresh_syllabus_versions(self) -> None:
        versions = self.database.list_syllabus_versions()
        self.syllabus_versions_list.clear()
        for version in versions:
            course = version["course_name"] or "Course not linked"
            code = f" ({version['course_code']})" if version["course_code"] else ""
            filename = Path(version["source_pdf_path"]).name
            item = QListWidgetItem(
                f"{course}{code} — {filename}\n"
                f"Version {version['version_number']} · "
                f"{version['review_status'].title()}"
            )
            item.setData(Qt.ItemDataRole.UserRole, version["id"])
            self.syllabus_versions_list.addItem(item)

    def _predict_page(self) -> QWidget:
        page, layout = self._new_page(
            "Classify a question",
            "Review suggested course outcome and cognitive-level labels before use.",
        )
        form_card = self._card()
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(22, 20, 22, 20)
        form_layout.setSpacing(12)

        course_label = QLabel("Course")
        course_label.setObjectName("fieldLabel")
        self.course_selector = QComboBox()
        self.course_selector.setPlaceholderText("Select a course")
        self.course_selector.setEnabled(False)
        self.course_selector.currentIndexChanged.connect(
            self._update_prediction_button
        )

        question_label = QLabel("Examination question")
        question_label.setObjectName("fieldLabel")
        self.question_input = QTextEdit()
        self.question_input.setPlaceholderText(
            "Enter an examination question to receive suggested labels."
        )
        self.question_input.setMinimumHeight(150)
        self.question_input.textChanged.connect(self._update_prediction_button)

        self.predict_button = QPushButton("Predict CO and BL")
        self.predict_button.setObjectName("primaryButton")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self._start_prediction)

        self.prediction_status = QLabel(
            "Suggestions are intended to support faculty review."
        )
        self.prediction_status.setObjectName("noticeText")
        self.prediction_status.setWordWrap(True)

        self.prediction_result = QLabel("")
        self.prediction_result.setObjectName("predictionResult")
        self.prediction_result.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.prediction_result.setWordWrap(True)

        form_layout.addWidget(course_label)
        form_layout.addWidget(self.course_selector)
        form_layout.addWidget(question_label)
        form_layout.addWidget(self.question_input)
        form_layout.addWidget(self.predict_button, 0, Qt.AlignmentFlag.AlignLeft)
        form_layout.addWidget(self.prediction_result)
        form_layout.addWidget(self.prediction_status)
        layout.addWidget(form_card)
        layout.addStretch(1)
        return page

    def _start_prediction(self) -> None:
        course_code = self.course_selector.currentData(Qt.ItemDataRole.UserRole)
        question = self.question_input.toPlainText()
        if not course_code or not question.strip():
            self._update_prediction_button()
            return
        self.prediction_result.clear()
        self.prediction_status.setText("Preparing a suggestion...")
        worker = PredictionWorker(
            self.database,
            course_code,
            question,
            self.bloom_config,
        )
        worker.completed.connect(self._prediction_completed)
        worker.failed.connect(self._prediction_failed)
        worker.cancelled.connect(self._prediction_cancelled)
        worker.finished.connect(self._prediction_worker_finished)
        self._prediction_worker = worker
        self._update_prediction_button()
        worker.start()

    def _prediction_completed(self, formatted_result: str) -> None:
        self.prediction_result.setText(formatted_result)
        self.prediction_status.setText(
            "Suggestion ready for faculty review."
        )
        self.refresh_question_history()

    def _prediction_failed(self, message: str) -> None:
        self.prediction_result.clear()
        self.prediction_status.setText(message)

    def _prediction_cancelled(self) -> None:
        if not self.prediction_result.text():
            self.prediction_status.setText("Classification was cancelled.")

    def _prediction_worker_finished(self) -> None:
        worker = self._prediction_worker
        self._prediction_worker = None
        self._update_prediction_button()
        if worker is not None:
            worker.deleteLater()

    def _history_page(self) -> QWidget:
        page, layout = self._new_page(
            "History", "Review questions and their suggested or approved labels."
        )
        toolbar = QHBoxLayout()
        self.history_search = QLineEdit()
        self.history_search.setPlaceholderText("Search questions or courses")
        self.history_search.textChanged.connect(self.refresh_question_history)
        toolbar.addWidget(self.history_search, 1)

        self.review_question_button = QPushButton("Review selected")
        self.review_question_button.setObjectName("secondaryButton")
        self.review_question_button.setEnabled(False)
        self.review_question_button.clicked.connect(self._review_selected_question)
        toolbar.addWidget(self.review_question_button)
        backup_button = QPushButton("Back up data")
        backup_button.setObjectName("secondaryButton")
        backup_button.clicked.connect(self._backup_data)
        export_button = QPushButton("Export data")
        export_button.setObjectName("secondaryButton")
        export_button.clicked.connect(self._export_data)
        toolbar.addWidget(backup_button)
        toolbar.addWidget(export_button)
        layout.addLayout(toolbar)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(
            ["Course", "Question", "Suggested CO / BL", "Approved CO / BL", "Review", "Date"]
        )
        self.history_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        self.history_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.history_table.itemSelectionChanged.connect(
            self._update_review_button
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        layout.addWidget(self.history_table, 1)
        self.history_empty_label = QLabel("No questions have been saved yet.")
        self.history_empty_label.setObjectName("mutedText")
        layout.addWidget(self.history_empty_label)
        self.refresh_question_history()
        layout.addStretch(1)
        return page

    def refresh_question_history(self, query: str | None = None) -> None:
        search_text = (
            self.history_search.text()
            if query is None and hasattr(self, "history_search")
            else query or ""
        )
        records = self.database.list_question_history(search_text)
        self.history_table.setRowCount(len(records))
        for row, record in enumerate(records):
            values = (
                f"{record['course_name']} ({record['course_code']})",
                record["question_text"],
                self._labels_text(record["predicted_co"], record["predicted_bl"]),
                self._labels_text(record["approved_co"], record["approved_bl"]),
                record["label_review_status"] or "Not reviewed",
                record["created_at"],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, record)
                self.history_table.setItem(row, column, item)
        self.history_empty_label.setVisible(not records)
        self._update_review_button()

    def _update_review_button(self) -> None:
        self.review_question_button.setEnabled(
            self.history_table.currentRow() >= 0
        )

    def _review_selected_question(self) -> None:
        row = self.history_table.currentRow()
        item = self.history_table.item(row, 0) if row >= 0 else None
        record = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not record:
            return

        dialog = LabelReviewDialog(
            question_text=record["question_text"],
            predicted_co=record["predicted_co"],
            predicted_bl=record["predicted_bl"],
            approved_co=record["approved_co"],
            approved_bl=record["approved_bl"],
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        review_status = (
            "corrected"
            if record["prediction_id"] is not None
            and (
                dialog.co_label != record["predicted_co"]
                or dialog.bl_label != record["predicted_bl"]
            )
            else "approved"
        )
        try:
            self.database.save_approved_labels(
                record["question_id"],
                dialog.co_label,
                dialog.bl_label,
                prediction_id=record["prediction_id"],
                review_status=review_status,
                reviewed_by=dialog.reviewed_by or None,
                review_note=dialog.review_note or None,
            )
        except (DatabaseError, ValueError) as error:
            QMessageBox.critical(self, "Review could not be saved", str(error))
            return
        self.refresh_question_history()

    @staticmethod
    def _labels_text(co_label: str | None, bl_label: str | None) -> str:
        if co_label is None and bl_label is None:
            return "—"
        return f"{co_label or 'N/A'} / {bl_label or 'N/A'}"

    def _backup_data(self) -> None:
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Back up application data",
            "co-bl-predictor-backup.sqlite3",
            "SQLite backup (*.sqlite3)",
        )
        if not destination:
            return
        try:
            saved_path = self.database.backup(destination)
        except (DatabaseError, OSError, ValueError) as error:
            QMessageBox.critical(self, "Backup failed", str(error))
            return
        QMessageBox.information(
            self,
            "Backup created",
            f"Your data backup was saved to:\n{saved_path}",
        )

    def _export_data(self) -> None:
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export application data",
            "co-bl-predictor-export.json",
            "JSON export (*.json)",
        )
        if not destination:
            return
        try:
            saved_path = self.database.export_json(destination)
        except (DatabaseError, OSError, ValueError) as error:
            QMessageBox.critical(self, "Export failed", str(error))
            return
        QMessageBox.information(
            self,
            "Export complete",
            f"Your data export was saved to:\n{saved_path}",
        )

    def close_storage(self) -> None:
        if self._prediction_worker is not None:
            self._prediction_worker.cancel()
            self._prediction_worker.wait()
        self.database.close()

    def _new_page(self, heading: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(16)

        title = QLabel(heading)
        title.setObjectName("sectionTitle")
        description = QLabel(subtitle)
        description.setObjectName("mutedText")
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)
        return page, layout

    @staticmethod
    def _card() -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        return card

    def _empty_state(self, title_text: str, body_text: str) -> QFrame:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 22, 22, 22)
        title = QLabel(title_text)
        title.setObjectName("cardTitle")
        body = QLabel(body_text)
        body.setObjectName("bodyText")
        body.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(body)
        return card

    def navigate(self, page_name: str) -> None:
        page = self.pages.get(page_name)
        if page is None:
            raise ValueError(f"Unknown application page: {page_name}")

        self.page_stack.setCurrentWidget(page)
        self.page_title.setText(page_name)
        for name, button in self.navigation_buttons.items():
            button.setChecked(name == page_name)

    @staticmethod
    def _stylesheet() -> str:
        return """
            QWidget {
                background: #f8f7fc;
                color: #292536;
                font-family: "Segoe UI";
                font-size: 14px;
            }
            QWidget#sidebar {
                background: #ffffff;
                border-right: 1px solid #e9e5f0;
            }
            QLabel#brand {
                color: #000000;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#sidebarCaption {
                color: #817b8d;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            QPushButton#navigationButton {
                background: transparent;
                border: 0;
                border-radius: 8px;
                color: #625c6e;
                min-height: 42px;
                padding: 0 12px;
                text-align: left;
            }
            QPushButton#navigationButton:hover {
                background: #f5f3ff;
                color: #7C3AED;
            }
            QPushButton#navigationButton:checked {
                background: #7C3AED;
                color: #ffffff;
                font-weight: 700;
            }
            QPushButton#navigationButton:checked:hover {
                background: #7C3AED;
                color: #ffffff;
                font-weight: 700;
            }
            QLabel#pageTitle {
                color: #292536;
                font-size: 25px;
                font-weight: 700;
            }
            QLabel#sectionTitle {
                color: #292536;
                font-size: 20px;
                font-weight: 700;
            }
            QFrame#card {
                background: #ffffff;
                border: 1px solid #e9e5f0;
                border-radius: 12px;
            }
            QLabel#cardTitle {
                color: #393346;
                font-size: 16px;
                font-weight: 700;
            }
            QLabel#bodyText {
                color: #625c6e;
                font-size: 14px;
            }
            QLabel#mutedText {
                color: #817b8d;
                font-size: 13px;
            }
            QLabel#fieldLabel {
                color: #494355;
                font-weight: 600;
            }
            QLabel#noticeText {
                background: #f5f3ff;
                border-radius: 8px;
                color: #5b21b6;
                padding: 10px;
            }
            QPushButton#primaryButton, QPushButton#secondaryButton {
                border-radius: 8px;
                font-weight: 600;
                min-height: 38px;
                padding: 0 16px;
            }
            QPushButton#primaryButton {
                background: #7C3AED;
                border: 1px solid #7C3AED;
                color: #ffffff;
            }
            QPushButton#primaryButton:hover:enabled {
                background: #6d28d9;
            }
            QPushButton#primaryButton:disabled {
                background: #c4b5fd;
                border-color: #c4b5fd;
                color: #ffffff;
            }
            QPushButton#secondaryButton {
                background: #ffffff;
                border: 1px solid #d8d1e5;
                color: #6d28d9;
            }
            QPushButton#secondaryButton:hover {
                background: #f5f3ff;
            }
            QComboBox, QLineEdit, QTextEdit {
                background: #ffffff;
                border: 1px solid #d8d1e5;
                border-radius: 7px;
                padding: 8px;
                selection-background-color: #7C3AED;
                selection-color: #ffffff;
            }
            QComboBox:disabled, QLineEdit:disabled, QTextEdit:disabled {
                color: #817b8d;
                background: #f4f2f7;
            }
            QStatusBar {
                background: #ffffff;
                border-top: 1px solid #e9e5f0;
                color: #625c6e;
            }
        """
