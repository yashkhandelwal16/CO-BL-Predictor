import logging
from collections.abc import Mapping, Sequence
from threading import Event

from PySide6.QtCore import QObject, QThread, Signal

from src.prediction.bloom_taxonomy import classify_bloom_level
from src.prediction.web_search_prediction import (
    WebSearchPredictionResult,
    WebSearchPredictionService,
)
from src.research.web_search import WebSearchError


_LOGGER = logging.getLogger(__name__)


class WebSearchPredictionWorker(QThread):
    completed = Signal(str, str)
    cancelled = Signal()

    def __init__(
        self,
        *,
        question: str,
        course_name: str,
        course_code: str,
        course_outcomes: Mapping[str, str],
        syllabus_units: (
            Mapping[str, Sequence[str]]
            | Sequence[Mapping[str, object]]
        ),
        service: WebSearchPredictionService | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._question = question
        self._course_name = course_name
        self._course_code = course_code
        self._course_outcomes = dict(course_outcomes)
        self._syllabus_units = syllabus_units
        self._service = service or WebSearchPredictionService()
        self._cancellation_event = Event()

    def cancel(self) -> None:
        self._cancellation_event.set()

    def run(self) -> None:
        try:
            result = self._service.predict(
                question=self._question,
                course_name=self._course_name,
                course_code=self._course_code,
                course_outcomes=self._course_outcomes,
                syllabus_units=self._syllabus_units,
                cancellation_event=self._cancellation_event,
            )
        except WebSearchError as error:
            result = self._unavailable_result(str(error))
        except Exception:
            _LOGGER.exception("Independent Web Search prediction failed.")
            result = self._unavailable_result(
                "Web search could not be completed. The existing suggestion "
                "is unaffected."
            )

        if self._cancellation_event.is_set():
            self.cancelled.emit()
            return
        self.completed.emit(result.format_output(), result.status_message)

    def _unavailable_result(
        self, status_message: str
    ) -> WebSearchPredictionResult:
        return WebSearchPredictionResult(
            co_label="N/A",
            bl_label=classify_bloom_level(self._question),
            status_message=status_message,
        )
