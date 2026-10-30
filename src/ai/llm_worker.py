from collections.abc import Sequence
from threading import Event

from PySide6.QtCore import QThread, Signal

from src.ai.llm_client import (
    ChatMessage,
    GenerationCancelledError,
    InferenceError,
    LLMClient,
    ModelConfigurationError,
)


class LLMWorker(QThread):
    """Run model loading and inference away from the GUI thread."""

    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        client: LLMClient,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 128,
        timeout_seconds: float | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._messages = list(messages)
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._cancellation_event = Event()

    def cancel(self) -> None:
        self._cancellation_event.set()

    def run(self) -> None:
        try:
            response = self._client.chat(
                self._messages,
                max_tokens=self._max_tokens,
                timeout_seconds=self._timeout_seconds,
                cancellation_event=self._cancellation_event,
            )
        except GenerationCancelledError:
            self.cancelled.emit()
        except ModelConfigurationError as error:
            self.failed.emit(str(error))
        except InferenceError as error:
            self.failed.emit(str(error))
        else:
            self.completed.emit(response)
