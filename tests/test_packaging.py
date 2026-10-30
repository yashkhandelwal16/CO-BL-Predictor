import sys

import pytest

from src.ai.llm_client import LLMClient, ModelLoadError, ModelNotFoundError
from src.database.db_manager import default_database_path
from src.prediction.bloom_rules import load_bloom_config, save_bloom_config
from src.utils.resource_paths import (
    application_data_root,
    application_resource_path,
    application_resource_root,
)


MODEL_FILENAME = "qwen2.5-0.5b-instruct-q4_k_m.gguf"


def test_packaged_resource_and_default_model_paths_use_pyinstaller_bundle(
    tmp_path, monkeypatch
):
    resources = tmp_path / "_internal"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(resources), raising=False)
    monkeypatch.delenv("CO_BL_MODEL_PATH", raising=False)

    assert application_resource_root() == resources
    assert application_resource_path("config", "bloom_levels.json") == (
        resources / "config" / "bloom_levels.json"
    )
    assert LLMClient().model_path == resources / "models" / MODEL_FILENAME


def test_packaged_bloom_rules_are_copied_to_writable_user_data(
    tmp_path, monkeypatch
):
    resources = tmp_path / "_internal"
    config_directory = resources / "config"
    config_directory.mkdir(parents=True)
    bundled_config = config_directory / "bloom_levels.json"
    bundled_config.write_text(
        (application_resource_root() / "config" / "bloom_levels.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    default_config = config_directory / "bloom_levels.default.json"
    default_config.write_text(
        (
            application_resource_root()
            / "config"
            / "bloom_levels.default.json"
        ).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(resources), raising=False)

    config = load_bloom_config()
    user_config = (
        local_app_data / "CO-BL-Predictor" / "config" / "bloom_levels.json"
    )
    assert user_config.is_file()
    original_user_content = user_config.read_text(encoding="utf-8")
    config["levels"][1]["keywords"] = ["package-specific"]
    save_bloom_config(config)

    assert "package-specific" in user_config.read_text(encoding="utf-8")
    assert bundled_config.read_text(encoding="utf-8") == original_user_content
    assert load_bloom_config()["levels"][1]["keywords"] == ["package-specific"]


def test_packaged_database_path_is_outside_bundle_and_uses_local_app_data(
    tmp_path, monkeypatch
):
    resources = tmp_path / "_internal"
    local_app_data = tmp_path / "LocalAppData"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(resources), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_app_data))

    database_path = default_database_path()

    assert database_path == (
        local_app_data / "CO-BL-Predictor" / "predictor.sqlite3"
    )
    assert resources not in database_path.parents
    assert application_data_root() == local_app_data / "CO-BL-Predictor"


def test_model_startup_validation_rejects_missing_or_invalid_assets(tmp_path):
    with pytest.raises(ModelNotFoundError, match="was not found"):
        LLMClient(tmp_path / "missing.gguf").validate_model_asset()

    invalid_model = tmp_path / "invalid.gguf"
    invalid_model.write_bytes(b"not a gguf file")
    with pytest.raises(ModelLoadError, match="not a valid GGUF"):
        LLMClient(invalid_model).validate_model_asset()
