from threading import Event
from typing import Any

from PySide6.QtCore import QThread, Signal

from src.ai.llm_client import (
    GenerationCancelledError,
    InferenceTimeoutError,
    LLMClient,
    ModelConfigurationError,
)
from src.database.db_manager import DatabaseError, DatabaseManager
from src.prediction.predictor import (
    PredictionInputError,
    PredictionModelError,
    QuestionPredictor,
)


class PredictionWorker(QThread):
    """Run local question classification without blocking the interface."""

    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        database: DatabaseManager,
        course_code: str,
        question_text: str,
        bloom_config: dict[str, Any],
    ) -> None:
        super().__init__()
        self._database = database
        self._course_code = course_code
        self._question_text = question_text
        self._bloom_config = bloom_config
        self._cancellation_event = Event()

    def cancel(self) -> None:
        self._cancellation_event.set()

    def run(self) -> None:
        try:
            predictor = QuestionPredictor(
                self._database,
                LLMClient(),
                self._bloom_config,
            )
            result = predictor.predict(
                self._course_code,
                self._question_text,
                cancellation_event=self._cancellation_event,
            )
        except (PredictionInputError, ModelConfigurationError) as error:
            self.failed.emit(str(error))
        except GenerationCancelledError:
            self.cancelled.emit()
        except InferenceTimeoutError:
            self.failed.emit(
                "Classification took too long. Try a shorter question and try again."
            )
        except (PredictionModelError, DatabaseError):
            self.failed.emit(
                "Classification could not be completed. Check the local "
                "application files and try again."
            )
        else:
            self.completed.emit(result.format_output())
