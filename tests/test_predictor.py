import importlib.util
import socket
from pathlib import Path

import pytest

from src.ai.llm_client import LLMClient
from src.courses.course_manager import COURSE_OUTCOME_IDS, CourseManager
from src.database.db_manager import DatabaseManager
from src.prediction.bloom_rules import load_bloom_config
from src.prediction.predictor import (
    PredictionInputError,
    QuestionPredictor,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / (
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)


class FakeChatClient:
    model_path = Path("local-test-model.gguf")

    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages, **options):
        self.messages.append((messages, options))
        return self.responses.pop(0)


def outcome_descriptions():
    return {
        outcome_id: f"Students will demonstrate outcome {number}."
        for number, outcome_id in enumerate(COURSE_OUTCOME_IDS, start=1)
    }


def make_database(tmp_path):
    database = DatabaseManager(tmp_path / "predictor.sqlite3")
    manager = CourseManager(database)
    manager.add_course(
        "Computer Networks", "CS305", outcome_descriptions()
    )
    return database


def reviewed_syllabus(course_code="CS305"):
    return {
        "course_name": "Computer Networks",
        "course_code": course_code,
        "source_file": "networks.pdf",
        "review_status": "reviewed",
        "course_outcomes": [],
        "units": [
            {
                "unit_number": 1,
                "unit_title": "Routing",
                "topics": ["Routing algorithms", "Shortest paths"],
            },
            {
                "unit_number": 2,
                "unit_title": "Data link layer",
                "topics": ["Framing", "Error detection"],
            },
        ],
    }


def test_prediction_formats_only_the_two_required_lines_and_persists_metadata(
    tmp_path,
):
    database = make_database(tmp_path)
    database.save_syllabus_version(
        reviewed_syllabus(), tmp_path / "networks.json"
    )
    client = FakeChatClient(["CO2", "BL3"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    result = predictor.predict("cs305", "Calculate a shortest routing path.")

    assert result.format_output() == "CO level-CO2\nBL level - BL3"
    assert result.status == "predicted"
    assert len(client.messages) == 2
    assert "CO1:" in client.messages[0][0][1]["content"]
    assert "CO4:" in client.messages[0][0][1]["content"]
    assert "Unit 1: Routing" in client.messages[0][0][1]["content"]
    assert "Unit 2: Data link layer" not in client.messages[0][0][1]["content"]
    assert "Never answer" in client.messages[0][0][0]["content"]
    assert "BL1 (Remember)" in client.messages[1][0][1]["content"]
    assert "Keyword matches are evidence only" in client.messages[1][0][0]["content"]

    history = database.list_question_history()
    assert len(history) == 1
    assert history[0]["predicted_co"] == "CO2"
    assert history[0]["predicted_bl"] == "BL3"
    with database._lock:
        prediction = database._connection.execute(
            "SELECT status, provider_name, model_name, model_version, "
            "syllabus_version_id FROM predictions WHERE id = ?",
            (result.prediction_id,),
        ).fetchone()
    assert tuple(prediction) == (
        "predicted",
        "Local",
        "local-test-model.gguf",
        QuestionPredictor._runtime_version(),
        result.syllabus_version_id,
    )
    assert result.syllabus_version_id is not None
    database.close()


@pytest.mark.parametrize(
    (
        "question_text",
        "co_response",
        "bl_response",
        "expected_co",
        "expected_bl",
    ),
    [
        ("Compare two routing strategies.", "N/A", "BL2", "N/A", "BL2"),
        ("Compare two routing strategies.", "CO3", "N/A", "CO3", "N/A"),
        ("How?", "N/A", "N/A", "N/A", "N/A"),
        ("Explain photosynthesis.", "N/A", "N/A", "N/A", "N/A"),
        ("Compare two routing strategies.", "The answer is CO1.", "BL4", "N/A", "BL4"),
        ("Compare two routing strategies.", "CO9", "BL8", "N/A", "N/A"),
        ("Compare two routing strategies.", "CO1\nBL2", "BL3", "N/A", "BL3"),
    ],
)
def test_uncertain_and_malformed_labels_fall_back_independently_to_na(
    tmp_path,
    question_text,
    co_response,
    bl_response,
    expected_co,
    expected_bl,
):
    database = make_database(tmp_path)
    client = FakeChatClient([co_response, bl_response])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    result = predictor.predict("CS305", question_text)

    assert result.format_output() == (
        f"CO level-{expected_co}\nBL level - {expected_bl}"
    )
    assert result.status == (
        "insufficient_context"
        if expected_co == "N/A" or expected_bl == "N/A"
        else "predicted"
    )
    database.close()


def test_empty_question_is_rejected_without_model_or_database_records(tmp_path):
    database = make_database(tmp_path)
    client = FakeChatClient(["CO1", "BL1"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    with pytest.raises(PredictionInputError, match="Enter an examination question"):
        predictor.predict("CS305", " \n\t ")

    assert client.messages == []
    assert database.list_question_history() == []
    database.close()


def test_missing_course_is_rejected_before_model_call(tmp_path):
    database = make_database(tmp_path)
    client = FakeChatClient(["CO1", "BL1"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    with pytest.raises(PredictionInputError, match="no longer available"):
        predictor.predict("MISSING", "Explain routing.")

    assert client.messages == []
    database.close()


def test_unlinked_syllabus_is_not_used_as_reviewed_course_context(tmp_path):
    database = make_database(tmp_path)
    unlinked = reviewed_syllabus("OTHER")
    database.save_syllabus_version(unlinked, tmp_path / "other.json")
    client = FakeChatClient(["CO1", "BL1"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    result = predictor.predict("CS305", "Explain network routing.")

    assert result.syllabus_version_id is None
    prompt = client.messages[0][0][1]["content"]
    assert "No relevant reviewed syllabus topics were found." in prompt
    assert "Routing algorithms" not in prompt
    database.close()


def test_prediction_prompts_include_only_relevant_faculty_reviewed_examples(
    tmp_path,
):
    database = make_database(tmp_path)
    reviewed_question = database.create_question(
        "CS305", "Explain how routers forward packets using routing tables."
    )
    reviewed_prediction = database.create_prediction(
        reviewed_question["id"], "CO2", "BL2"
    )
    database.save_approved_labels(
        reviewed_question["id"],
        "CO3",
        "BL3",
        prediction_id=reviewed_prediction["id"],
        review_status="corrected",
    )
    unreviewed_question = database.create_question(
        "CS305", "Describe router packet forwarding with a routing table."
    )
    database.create_prediction(
        unreviewed_question["id"], "CO1", "BL1"
    )
    client = FakeChatClient(["CO3", "BL3"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    result = predictor.predict(
        "CS305", "Explain how routers forward packets using routing table."
    )

    assert result.co_label == "CO3"
    for messages, _ in client.messages:
        prompt = messages[1]["content"]
        assert "Explain how routers forward packets using routing tables." in prompt
        assert "Approved labels: CO3, BL3" in prompt
        assert "Describe router packet forwarding with a routing table." not in prompt
    database.close()


def test_empty_model_response_is_rejected_as_uncertain_label(tmp_path):
    database = make_database(tmp_path)
    client = FakeChatClient(["", "BL2"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    result = predictor.predict("CS305", "Describe the network layers.")

    assert result.co_label == "N/A"
    assert result.bl_label == "BL2"
    database.close()


@pytest.mark.skipif(
    not MODEL_PATH.is_file() or importlib.util.find_spec("llama_cpp") is None,
    reason="Qwen GGUF and llama-cpp-python are both required",
)
def test_real_model_prediction_has_exactly_two_valid_output_lines(
    tmp_path, monkeypatch
):
    def block_connection(*args, **kwargs):
        raise AssertionError("Local prediction must not open network connections.")

    monkeypatch.setattr(socket, "create_connection", block_connection)
    monkeypatch.setattr(socket.socket, "connect", block_connection)
    database = make_database(tmp_path)
    predictor = QuestionPredictor(
        database,
        LLMClient(MODEL_PATH, context_length=2048, threads=4),
        load_bloom_config(),
    )

    result = predictor.predict(
        "CS305",
        "Describe how a router forwards packets using a routing table.",
    )
    output_lines = result.format_output().splitlines()

    assert len(output_lines) == 2
    assert output_lines[0] in {
        "CO level-CO1",
        "CO level-CO2",
        "CO level-CO3",
        "CO level-CO4",
        "CO level-N/A",
    }
    assert output_lines[1] in {
        "BL level - BL0",
        "BL level - BL1",
        "BL level - BL2",
        "BL level - BL3",
        "BL level - BL4",
        "BL level - N/A",
    }
    history = database.list_question_history()
    assert len(history) == 1
    assert history[0]["predicted_co"] == result.co_label
    assert history[0]["predicted_bl"] == result.bl_label
    assert history[0]["provider_name"] == "Local"
    assert history[0]["model_name"] == MODEL_PATH.name
    assert "router" not in result.format_output().casefold()
    database.close()
