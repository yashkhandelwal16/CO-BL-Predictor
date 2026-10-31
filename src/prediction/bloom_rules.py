import json
import re
import shutil
from pathlib import Path
from typing import Any

from src.utils.resource_paths import (
    application_data_root,
    application_resource_path,
    is_packaged_application,
)


EXPECTED_LEVEL_IDS = ("BL0", "BL1", "BL2", "BL3", "BL4", "BL5")
LEGACY_LEVEL_IDS = ("BL0", "BL1", "BL2", "BL3", "BL4")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "bloom_levels.default.json"
_CONFIG_PATH = _PROJECT_ROOT / "config" / "bloom_levels.json"


class BloomConfigurationError(ValueError):
    """Raised when Bloom-style level configuration is missing or invalid."""


def load_bloom_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path is not None else _active_config_path()
    if path is None and is_packaged_application():
        _initialize_user_config(config_path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise BloomConfigurationError(
            f"Classification configuration was not found: {config_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise BloomConfigurationError(
            f"Classification configuration is not valid JSON: {error}"
        ) from error
    except OSError as error:
        raise BloomConfigurationError(
            f"Classification configuration could not be read: {error}"
        ) from error

    config = _upgrade_legacy_config(config)
    validate_bloom_config(config)
    return config


def _upgrade_legacy_config(config: Any) -> Any:
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != 1
        or config.get("classification_scheme") != "custom_bl0_bl4"
    ):
        return config
    levels = config.get("levels")
    if not isinstance(levels, list) or tuple(
        level.get("id") if isinstance(level, dict) else None for level in levels
    ) != LEGACY_LEVEL_IDS:
        return config

    config["classification_scheme"] = "custom_bl0_bl5"
    if config["levels"][0].get("description") == (
        "The question cannot be classified under the configured BL1-BL4 "
        "scheme despite sufficient context."
    ):
        config["levels"][0]["description"] = (
            "The question cannot be classified under the configured BL1-BL5 "
            "scheme despite sufficient context."
        )
    config["levels"].append(
        {
            "id": "BL5",
            "name": "Evaluate",
            "description": (
                "Make and justify a judgment using criteria, evidence, or "
                "standards; assess alternatives, validity, or quality."
            ),
            "keywords": [
                "evaluate",
                "assess",
                "justify",
                "critique",
                "recommend",
                "defend",
                "judge",
            ],
        }
    )
    return config


def validate_bloom_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise BloomConfigurationError("Configuration must be a JSON object.")
    if config.get("schema_version") != 1:
        raise BloomConfigurationError("Unsupported classification configuration version.")
    if config.get("classification_scheme") != "custom_bl0_bl5":
        raise BloomConfigurationError("The configured classification scheme is invalid.")
    if config.get("definitions_status") not in {
        "draft_unverified",
        "institution_approved",
    }:
        raise BloomConfigurationError(
            "definitions_status must be draft_unverified or institution_approved."
        )

    levels = config.get("levels")
    if not isinstance(levels, list):
        raise BloomConfigurationError("levels must be a list.")
    ids = [level.get("id") if isinstance(level, dict) else None for level in levels]
    if tuple(ids) != EXPECTED_LEVEL_IDS:
        raise BloomConfigurationError(
            "Levels must appear exactly once and in order: BL0, BL1, BL2, BL3, BL4."
        )

    for level in levels:
        level_id = level["id"]
        for field in ("name", "description"):
            value = level.get(field)
            if not isinstance(value, str) or not value.strip():
                raise BloomConfigurationError(
                    f"{level_id} must have a non-empty {field}."
                )

        keywords = level.get("keywords")
        if not isinstance(keywords, list) or not all(
            isinstance(keyword, str) and keyword.strip()
            for keyword in keywords
        ):
            raise BloomConfigurationError(
                f"{level_id} keywords must be a list of non-empty strings."
            )
        normalized_keywords = [keyword.strip().casefold() for keyword in keywords]
        if len(normalized_keywords) != len(set(normalized_keywords)):
            raise BloomConfigurationError(
                f"{level_id} contains duplicate keywords."
            )
    if levels[0]["keywords"]:
        raise BloomConfigurationError("BL0 must not have matching keywords.")


def save_bloom_config(
    config: dict[str, Any], path: str | Path | None = None
) -> Path:
    validate_bloom_config(config)
    config_path = Path(path) if path is not None else _active_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config_path.with_suffix(config_path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(config_path)
    except OSError:
        temporary_path.unlink(missing_ok=True)
        raise
    return config_path


def restore_default_bloom_config(
    path: str | Path | None = None,
) -> dict[str, Any]:
    defaults = load_bloom_config(
        application_resource_path("config", "bloom_levels.default.json")
    )
    save_bloom_config(defaults, path)
    return defaults


def _active_config_path() -> Path:
    if is_packaged_application():
        return application_data_root() / "config" / "bloom_levels.json"
    return _CONFIG_PATH


def _initialize_user_config(config_path: Path) -> None:
    if config_path.exists():
        return
    bundled_path = application_resource_path("config", "bloom_levels.json")
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(bundled_path, config_path)
    except OSError as error:
        raise BloomConfigurationError(
            f"Classification configuration could not be initialized: {error}"
        ) from error


def collect_bloom_keyword_evidence(
    question: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    active_config = config if config is not None else load_bloom_config()
    validate_bloom_config(active_config)

    evidence = []
    for level in active_config["levels"]:
        matches = [
            keyword
            for keyword in level["keywords"]
            if _contains_whole_keyword(question, keyword)
        ]
        evidence.append(
            {
                "level": level["id"],
                "name": level["name"],
                "matched_keywords": matches,
            }
        )

    return {
        "question_has_text": bool(question.strip()),
        "definitions_status": active_config["definitions_status"],
        "levels": evidence,
    }


def _contains_whole_keyword(text: str, keyword: str) -> bool:
    pattern = rf"(?<!\w){re.escape(keyword.strip())}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None
