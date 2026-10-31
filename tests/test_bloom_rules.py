import copy
import json

import pytest

from src.prediction.bloom_rules import (
    BloomConfigurationError,
    collect_bloom_keyword_evidence,
    load_bloom_config,
    restore_default_bloom_config,
    save_bloom_config,
    validate_bloom_config,
)


def test_default_config_has_custom_levels_and_is_marked_unverified() -> None:
    config = load_bloom_config()

    assert [level["id"] for level in config["levels"]] == [
        "BL0",
        "BL1",
        "BL2",
        "BL3",
        "BL4",
        "BL5",
    ]
    assert config["classification_scheme"] == "custom_bl0_bl5"
    assert config["definitions_status"] == "draft_unverified"


def test_keyword_evidence_is_case_insensitive_and_matches_whole_words() -> None:
    config = load_bloom_config()

    evidence = collect_bloom_keyword_evidence(
        "EXPLAIN the statement and compare it with another statement.",
        config,
    )
    by_level = {item["level"]: item["matched_keywords"] for item in evidence["levels"]}

    assert by_level["BL2"] == ["explain"]
    assert by_level["BL4"] == ["compare"]
    assert by_level["BL5"] == []
    assert all("state" not in matches for matches in by_level.values())


def test_multiple_verb_levels_are_reported_without_selecting_a_prediction() -> None:
    evidence = collect_bloom_keyword_evidence(
        "Define the terms, explain the process, and calculate the result."
    )
    matched_levels = {
        item["level"] for item in evidence["levels"] if item["matched_keywords"]
    }

    assert matched_levels == {"BL1", "BL2", "BL3"}
    assert "prediction" not in evidence
    assert "label" not in evidence


def test_no_match_does_not_assign_bl0_or_n_a_automatically() -> None:
    evidence = collect_bloom_keyword_evidence(
        "A question with no configured cognitive verb."
    )

    assert evidence["question_has_text"]
    assert all(not item["matched_keywords"] for item in evidence["levels"])
    assert "prediction" not in evidence


def test_empty_question_has_no_keyword_evidence() -> None:
    evidence = collect_bloom_keyword_evidence(" \n ")

    assert not evidence["question_has_text"]
    assert all(not item["matched_keywords"] for item in evidence["levels"])


def test_changed_keywords_change_evidence_without_code_changes(tmp_path) -> None:
    config = load_bloom_config()
    config["levels"][1]["keywords"] = ["enumerate"]
    config_path = tmp_path / "bloom_levels.json"

    save_bloom_config(config, config_path)
    reloaded = load_bloom_config(config_path)
    evidence = collect_bloom_keyword_evidence("Enumerate three properties.", reloaded)
    by_level = {item["level"]: item["matched_keywords"] for item in evidence["levels"]}

    assert by_level["BL1"] == ["enumerate"]
    assert not by_level["BL2"]


def test_saving_invalid_config_is_rejected_without_writing(tmp_path) -> None:
    config = load_bloom_config()
    config["levels"][0]["keywords"] = ["unknown"]
    config_path = tmp_path / "bloom_levels.json"

    with pytest.raises(BloomConfigurationError, match="BL0"):
        save_bloom_config(config, config_path)
    assert not config_path.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data.update(schema_version=9), "Unsupported"),
        (lambda data: data["levels"].pop(), "exactly once"),
        (lambda data: data["levels"][1].update(name=" "), "non-empty name"),
        (
            lambda data: data["levels"][1].update(keywords=["Define", "define"]),
            "duplicate keywords",
        ),
    ],
)
def test_invalid_config_structures_are_rejected(mutation, message) -> None:
    config = copy.deepcopy(load_bloom_config())
    mutation(config)

    with pytest.raises(BloomConfigurationError, match=message):
        validate_bloom_config(config)


def test_malformed_or_missing_config_reports_error(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(BloomConfigurationError, match="not found"):
        load_bloom_config(missing_path)

    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text("{", encoding="utf-8")
    with pytest.raises(BloomConfigurationError, match="not valid JSON"):
        load_bloom_config(malformed_path)


def test_restore_defaults_replaces_custom_keywords(tmp_path) -> None:
    config_path = tmp_path / "bloom_levels.json"
    config = load_bloom_config()
    config["levels"][1]["keywords"] = ["enumerate"]
    save_bloom_config(config, config_path)

    restored = restore_default_bloom_config(config_path)

    assert restored == load_bloom_config()
    assert load_bloom_config(config_path) == restored
    assert "define" in restored["levels"][1]["keywords"]


def test_legacy_bl0_bl4_configuration_is_upgraded_with_evaluate(tmp_path) -> None:
    config = load_bloom_config()
    config["levels"] = config["levels"][:-1]
    config["classification_scheme"] = "custom_bl0_bl4"
    config["levels"][0]["description"] = (
        "The question cannot be classified under the configured BL1-BL4 "
        "scheme despite sufficient context."
    )
    legacy_path = tmp_path / "legacy-bloom.json"
    legacy_path.write_text(
        json.dumps(config),
        encoding="utf-8",
    )

    upgraded = load_bloom_config(legacy_path)

    assert upgraded["classification_scheme"] == "custom_bl0_bl5"
    assert [level["id"] for level in upgraded["levels"]][-1] == "BL5"
    assert upgraded["levels"][-1]["name"] == "Evaluate"
