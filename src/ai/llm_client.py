import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from threading import Event, Lock
from typing import Any, Literal, Protocol, TypedDict


ChatRole = Literal["system", "user", "assistant"]


class ChatMessage(TypedDict):
    role: ChatRole
    content: str


class _ChatModel(Protocol):
    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        stream: bool,
    ) -> Iterator[dict[str, Any]]: ...


class ModelConfigurationError(ValueError):
    """Raised when a local inference setting is invalid."""


class InferenceError(RuntimeError):
    """Raised when local text generation fails."""


class ModelNotFoundError(InferenceError):
    """Raised when the configured local model file is missing."""


class ModelRuntimeUnavailableError(InferenceError):
    """Raised when the local inference package cannot be imported."""


class ModelLoadError(InferenceError):
    """Raised when the local model cannot be loaded."""


class GenerationCancelledError(InferenceError):
    """Raised when generation is cancelled by its caller."""


class InferenceTimeoutError(InferenceError):
    """Raised when generation exceeds its cooperative timeout."""


def default_model_path() -> Path:
    configured_path = os.environ.get("CO_BL_MODEL_PATH")
    if configured_path:
        return Path(configured_path).expanduser()
    return (
        Path(__file__).resolve().parents[2]
        / "models"
        / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
    )


def _read_positive_integer_setting(
    name: str, default: int, maximum: int | None = None
) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ModelConfigurationError(
            f"{name} must be a positive whole number."
        ) from error
    if value < 1 or (maximum is not None and value > maximum):
        bound = f" between 1 and {maximum}" if maximum else ""
        raise ModelConfigurationError(f"{name} must be{bound or ' positive'}.")
    return value


def _default_thread_count() -> int:
    return min(8, max(1, os.cpu_count() or 1))


class LLMClient:
    """Load and run a local GGUF chat model without making network requests."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        context_length: int | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_path = (
            Path(model_path).expanduser() if model_path else default_model_path()
        )
        if context_length is None:
            context_length = _read_positive_integer_setting(
                "CO_BL_MODEL_CONTEXT", 2048, maximum=32768
            )
        if threads is None:
            threads = _read_positive_integer_setting(
                "CO_BL_MODEL_THREADS", _default_thread_count()
            )
        if not 1 <= context_length <= 32768:
            raise ModelConfigurationError(
                "Model context length must be between 1 and 32768 tokens."
            )
        if threads < 1:
            raise ModelConfigurationError("Model thread count must be positive.")
        self.context_length = context_length
        self.threads = threads
        self._model: _ChatModel | None = None
        self._load_lock = Lock()
        self._inference_lock = Lock()

    def load(self) -> None:
        """Load the model once; callers should use a worker thread in the UI."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self.model_path.is_file():
                raise ModelNotFoundError(
                    f"Local model file was not found: {self.model_path}"
                )
            try:
                with self.model_path.open("rb") as model_file:
                    signature = model_file.read(4)
            except OSError as error:
                raise ModelLoadError(
                    f"Local model file could not be read: {error}"
                ) from error
            if signature != b"GGUF":
                raise ModelLoadError(
                    "The selected file is not a valid GGUF model. "
                    "Download the .gguf model file and try again."
                )

            model_factory = self._model_factory()
            try:
                self._model = model_factory(
                    model_path=str(self.model_path),
                    n_ctx=self.context_length,
                    n_threads=self.threads,
                    verbose=False,
                )
            except Exception as error:
                raise ModelLoadError(
                    "The local model could not be loaded. The file may be "
                    "incomplete, corrupt, or incompatible with this runtime."
                ) from error

    @staticmethod
    def _model_factory() -> Callable[..., _ChatModel]:
        try:
            from llama_cpp import Llama
        except ImportError as error:
            raise ModelRuntimeUnavailableError(
                "The local inference runtime is not installed. Install the "
                "project's llama-cpp-python dependency and restart."
            ) from error
        return Llama

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        timeout_seconds: float | None = None,
        cancellation_event: Event | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Generate local chat text, checking cancellation between output tokens.

        Timeouts and cancellation are cooperative: llama.cpp cannot be
        interrupted while evaluating the prompt or a single native token.
        """
        normalized_messages = self._validate_messages(messages)
        if max_tokens < 1:
            raise ModelConfigurationError("max_tokens must be positive.")
        if not 0.0 <= temperature <= 2.0:
            raise ModelConfigurationError("temperature must be between 0 and 2.")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ModelConfigurationError("timeout_seconds must be positive.")

        started_at = time.monotonic()
        deadline = (
            started_at + timeout_seconds if timeout_seconds is not None else None
        )
        self._check_cancelled(cancellation_event)
        self._acquire_inference_lock(deadline, cancellation_event)
        try:
            self._check_cancelled(cancellation_event)
            self.load()
            self._check_cancelled(cancellation_event)
            self._check_deadline(deadline)
            model = self._model
            if model is None:
                raise ModelLoadError("The local model did not finish loading.")
            try:
                chunks = model.create_chat_completion(
                    messages=normalized_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True,
                )
                return self._collect_stream(
                    chunks,
                    deadline=deadline,
                    cancellation_event=cancellation_event,
                    on_token=on_token,
                )
            except InferenceError:
                raise
            except Exception as error:
                raise InferenceError(
                    "Local text generation failed. Check the model file and try "
                    "again."
                ) from error
        finally:
            self._inference_lock.release()

    def _acquire_inference_lock(
        self, deadline: float | None, cancellation_event: Event | None
    ) -> None:
        while not self._inference_lock.acquire(timeout=0.1):
            self._check_cancelled(cancellation_event)
            self._check_deadline(deadline)

    @staticmethod
    def _check_cancelled(cancellation_event: Event | None) -> None:
        if cancellation_event is not None and cancellation_event.is_set():
            raise GenerationCancelledError("Generation was cancelled.")

    @staticmethod
    def _check_deadline(deadline: float | None) -> None:
        if deadline is not None and time.monotonic() >= deadline:
            raise InferenceTimeoutError("Local generation exceeded its time limit.")

    @classmethod
    def _collect_stream(
        cls,
        chunks: Iterator[dict[str, Any]],
        *,
        deadline: float | None,
        cancellation_event: Event | None,
        on_token: Callable[[str], None] | None,
    ) -> str:
        output: list[str] = []
        close = getattr(chunks, "close", None)
        try:
            for chunk in chunks:
                cls._check_cancelled(cancellation_event)
                cls._check_deadline(deadline)
                try:
                    token = chunk["choices"][0]["delta"].get("content") or ""
                except (KeyError, IndexError, TypeError) as error:
                    raise InferenceError(
                        "The local inference runtime returned an invalid response."
                    ) from error
                if not isinstance(token, str):
                    raise InferenceError(
                        "The local inference runtime returned non-text content."
                    )
                if token:
                    output.append(token)
                    if on_token is not None:
                        on_token(token)
            cls._check_cancelled(cancellation_event)
            cls._check_deadline(deadline)
        finally:
            if close is not None:
                close()
        response = "".join(output).strip()
        if not response:
            raise InferenceError("The local model returned an empty response.")
        return response

    @staticmethod
    def _validate_messages(
        messages: Sequence[ChatMessage],
    ) -> list[dict[str, str]]:
        if not messages:
            raise ModelConfigurationError("At least one chat message is required.")
        normalized: list[dict[str, str]] = []
        allowed_roles = {"system", "user", "assistant"}
        for message in messages:
            if (
                not isinstance(message, Mapping)
                or message.get("role") not in allowed_roles
                or not isinstance(message.get("content"), str)
                or not message["content"].strip()
            ):
                raise ModelConfigurationError(
                    "Each chat message must have a supported role and non-empty text."
                )
            normalized.append(
                {"role": str(message["role"]), "content": message["content"]}
            )
        return normalized
