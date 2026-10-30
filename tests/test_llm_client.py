import importlib.util
import socket
import threading
import time
from pathlib import Path

import pytest

from src.ai.llm_client import (
    GenerationCancelledError,
    InferenceTimeoutError,
    LLMClient,
    ModelConfigurationError,
    ModelLoadError,
    ModelNotFoundError,
)
from src.ai.llm_worker import LLMWorker


class FakeModel:
    def __init__(self, chunks):
        self.chunks = chunks
        self.call_options = None

    def create_chat_completion(self, **options):
        self.call_options = options
        return iter(self.chunks)


def valid_message():
    return [{"role": "user", "content": "Reply with a short greeting."}]


def make_client(tmp_path, monkeypatch, chunks=None):
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF" + b"\x00" * 12)
    model = FakeModel(
        chunks
        if chunks is not None
        else [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " there."}}]},
        ]
    )
    factory_options = {}

    def create_model(**options):
        factory_options.update(options)
        return model

    monkeypatch.setattr(
        LLMClient,
        "_model_factory",
        staticmethod(lambda: create_model),
    )
    return LLMClient(model_path, context_length=1024, threads=2), model, factory_options


def test_load_uses_configured_path_context_and_cpu_threads(tmp_path, monkeypatch):
    client, _, factory_options = make_client(tmp_path, monkeypatch)
    client.load()
    assert client._model is not None
    assert factory_options == {
        "model_path": str(client.model_path),
        "n_ctx": 1024,
        "n_threads": 2,
        "verbose": False,
    }


def test_missing_model_has_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="was not found"):
        LLMClient(tmp_path / "missing.gguf").load()


def test_rejects_non_gguf_and_invalid_configuration(tmp_path):
    invalid_model = tmp_path / "wrong.gguf"
    invalid_model.write_bytes(b"not a model")

    with pytest.raises(ModelLoadError, match="not a valid GGUF"):
        LLMClient(invalid_model).load()
    with pytest.raises(ModelConfigurationError, match="between 1 and 32768"):
        LLMClient(tmp_path / "missing.gguf", context_length=32769)
    with pytest.raises(ModelConfigurationError, match="thread count"):
        LLMClient(tmp_path / "missing.gguf", threads=0)


def test_chat_streams_local_response_and_supplies_cpu_settings(
    tmp_path, monkeypatch
):
    client, model, _ = make_client(tmp_path, monkeypatch)
    received_tokens = []

    response = client.chat(valid_message(), on_token=received_tokens.append)

    assert response == "Hello there."
    assert received_tokens == ["Hello", " there."]
    assert model.call_options == {
        "messages": valid_message(),
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": True,
    }


def test_cancelled_generation_stops_between_tokens(tmp_path, monkeypatch):
    cancellation = threading.Event()
    client, _, _ = make_client(tmp_path, monkeypatch)

    with pytest.raises(GenerationCancelledError):
        client.chat(
            valid_message(),
            cancellation_event=cancellation,
            on_token=lambda _token: cancellation.set(),
        )


def test_generation_timeout_is_checked_between_tokens(tmp_path, monkeypatch):
    client, _, _ = make_client(
        tmp_path,
        monkeypatch,
        chunks=[
            {"choices": [{"delta": {"content": "first"}}]},
            {"choices": [{"delta": {"content": "second"}}]},
        ],
    )

    with pytest.raises(InferenceTimeoutError):
        client.chat(
            valid_message(),
            timeout_seconds=0.01,
            on_token=lambda _token: time.sleep(0.02),
        )


def test_local_client_does_not_open_network_connections(
    tmp_path, monkeypatch
):
    client, _, _ = make_client(tmp_path, monkeypatch)

    def block_connection(*args, **kwargs):
        raise AssertionError("Local inference must not open network connections.")

    monkeypatch.setattr(socket, "create_connection", block_connection)
    monkeypatch.setattr(socket.socket, "connect", block_connection)
    response = client.chat(valid_message())

    assert response == "Hello there."


def test_default_model_path_can_be_overridden_by_environment(
    tmp_path, monkeypatch
):
    configured = tmp_path / "custom.gguf"
    monkeypatch.setenv("CO_BL_MODEL_PATH", str(configured))

    assert LLMClient().model_path == configured


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / (
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)
RUNTIME_INSTALLED = importlib.util.find_spec("llama_cpp") is not None


@pytest.mark.skipif(
    not MODEL_PATH.is_file() or not RUNTIME_INSTALLED,
    reason="Qwen GGUF and llama-cpp-python are both required",
)
def test_real_model_loads_and_handles_two_basic_classification_prompts(
    monkeypatch,
):
    monkeypatch.setattr(socket, "create_connection", _blocked_network_connection)
    monkeypatch.setattr(socket.socket, "connect", _blocked_network_connection)
    client = LLMClient(MODEL_PATH, context_length=2048, threads=4)
    prompts = (
        ("Choose the greater number: 9 or 2. Reply with only the number.", "9"),
        ("Choose the smaller number: 9 or 2. Reply with only the number.", "2"),
    )

    for prompt, expected in prompts:
        response = client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=8,
            timeout_seconds=120,
        )
        assert response.strip() == expected


def _blocked_network_connection(*args, **kwargs):
    raise AssertionError("Local inference must not open network connections.")


def test_worker_runs_model_call_off_the_calling_thread(tmp_path, monkeypatch):
    client, model, _ = make_client(tmp_path, monkeypatch)
    calling_thread = threading.get_ident()
    model_thread = []
    original_create = model.create_chat_completion

    def record_model_thread(**options):
        model_thread.append(threading.get_ident())
        return original_create(**options)

    model.create_chat_completion = record_model_thread
    client._model = model
    worker = LLMWorker(client, valid_message())
    worker.start()

    assert worker.wait(5000)
    assert model_thread and model_thread[0] != calling_thread
