from threading import Event
from typing import Sequence

from PySide6.QtCore import QThread, Signal

from src.research.web_content_extraction import (
    WebContentExtractionService,
    WebContentResult,
)
from src.research.web_search import SearchResult


class WebContentExtractionWorker(QThread):
    """Extract a bounded set of search results away from the GUI thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        service: WebContentExtractionService,
        results: Sequence[SearchResult],
    ) -> None:
        super().__init__()
        self._service = service
        self._results = tuple(results)
        self._cancellation_event = Event()

    def cancel(self) -> None:
        self._cancellation_event.set()

    def run(self) -> None:
        try:
            results: list[WebContentResult] = self._service.extract_results(
                self._results,
                cancellation_event=self._cancellation_event,
            )
        except (TypeError, ValueError):
            self.failed.emit("Webpage extraction could not be started.")
        except Exception:
            self.failed.emit("Webpage extraction could not be completed.")
        else:
            self.completed.emit(results)
