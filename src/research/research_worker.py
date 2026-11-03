import logging
from threading import Event
from typing import Mapping, Sequence

from PySide6.QtCore import QThread, Signal

from src.research.research_engine import ResearchEngine, ResearchEngineResult


_LOGGER = logging.getLogger(__name__)


class ResearchWorker(QThread):
    """Run bounded CO-specific web research away from the GUI thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        engine: ResearchEngine,
        *,
        academic_question: str,
        course_name: str,
        course_code: str,
        course_outcomes: Mapping[str, str],
        unit_topics: (
            Mapping[str, Sequence[str]]
            | Sequence[Mapping[str, object]]
        ),
        relevant_technical_concepts: Sequence[str],
        detected_cognitive_action: str,
    ) -> None:
        super().__init__()
        self._engine = engine
        self._inputs = {
            "academic_question": academic_question,
            "course_name": course_name,
            "course_code": course_code,
            "course_outcomes": dict(course_outcomes),
            "unit_topics": unit_topics,
            "relevant_technical_concepts": (
                relevant_technical_concepts
                if isinstance(relevant_technical_concepts, (str, bytes))
                else tuple(relevant_technical_concepts)
            ),
            "detected_cognitive_action": detected_cognitive_action,
        }
        self._cancellation_event = Event()

    def cancel(self) -> None:
        self._cancellation_event.set()

    def run(self) -> None:
        try:
            result: ResearchEngineResult = self._engine.research(
                **self._inputs,
                cancellation_event=self._cancellation_event,
            )
        except (TypeError, ValueError):
            _LOGGER.exception("Web research worker received invalid input.")
            self.failed.emit("Web research could not be started.")
        except Exception:
            _LOGGER.exception("Web research worker failed.")
            self.failed.emit("Web research could not be completed.")
        else:
            self.completed.emit(result)
