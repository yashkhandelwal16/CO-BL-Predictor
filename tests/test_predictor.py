import importlib.util
import re
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
        "CO1": "Explain computer network models and protocols.",
        "CO2": (
            "Analyze network routing algorithms, Dijkstra, and shortest paths."
        ),
        "CO3": "Apply data link layer framing and error detection.",
        "CO4": "Evaluate network security and performance.",
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


def test_prediction_by_course_id_uses_latest_reviewed_syllabus(tmp_path) -> None:
    database = make_database(tmp_path)
    first_version = database.save_syllabus_version(
        reviewed_syllabus(), tmp_path / "first.json"
    )
    updated_data = reviewed_syllabus()
    updated_data["units"] = [
        {
            "unit_number": 1,
            "unit_title": "Updated routing",
            "topics": ["Routing updates"],
        }
    ]
    latest_version = database.save_syllabus_version(
        updated_data, tmp_path / "latest.json"
    )
    course = database.get_course("CS305")
    assert course is not None
    predictor = QuestionPredictor(
        database,
        FakeChatClient(["CO2", "BL3"]),
        load_bloom_config(),
    )

    result = predictor.predict(
        course["id"], "Calculate a shortest routing path."
    )

    assert result.syllabus_version_id == latest_version["id"]
    assert result.syllabus_version_id != first_version["id"]
    assert "Updated routing" in predictor.llm_client.messages[0][0][1]["content"]
    history = database.list_question_history()
    assert history[0]["course_id"] == course["id"]
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


@pytest.mark.parametrize(
    ("question_text", "expected_message"),
    [
        (
            "Yash Khandelwal",
            "Please enter a valid academic question related to the selected course syllabus.",
        ),
        ("Hello", "This is not an academic question."),
        ("Good morning", "This is not an academic question."),
        ("How are you?", "This is not an academic question."),
        ("Hello, how are you?", "This is not an academic question."),
        ("What is your name?", "This is not an academic question."),
        (
            "Who won the cricket match yesterday?",
            "Please enter a valid academic question related to the selected course syllabus.",
        ),
        (
            "asdf qwer zxcv",
            "Please enter a valid academic question related to the selected course syllabus.",
        ),
    ],
)
def test_non_academic_and_unrelated_questions_are_rejected_before_inference(
    tmp_path, question_text, expected_message
):
    database = make_database(tmp_path)
    client = FakeChatClient(["CO1", "BL1"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    with pytest.raises(PredictionInputError, match=re.escape(expected_message)):
        predictor.predict("CS305", question_text)

    assert client.messages == []
    assert database.list_question_history() == []
    database.close()


@pytest.mark.parametrize(
    "question_text",
    ["Define inheritance in Java.", "Explain inheritance."],
)
def test_syllabus_related_academic_question_reaches_prediction(
    tmp_path, question_text
):
    database = DatabaseManager(tmp_path / "predictor.sqlite3")
    outcomes = {
        "CO1": "Explain classes and objects in Java.",
        "CO2": "Explain polymorphism and dynamic dispatch in Java.",
        "CO3": "Apply encapsulation and abstraction.",
        "CO4": "Analyze object-oriented design patterns.",
    }
    CourseManager(database).add_course(
        "Object Oriented Programming", "CS301", outcomes
    )
    syllabus = reviewed_syllabus("CS301")
    syllabus["course_name"] = "Object Oriented Programming"
    syllabus["units"] = [
        {
            "unit_number": 1,
            "unit_title": "Inheritance",
            "topics": ["Java inheritance"],
        }
    ]
    database.save_syllabus_version(syllabus, tmp_path / "oop.json")
    client = FakeChatClient(["CO2", "BL1"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    result = predictor.predict("CS301", question_text)

    assert result.co_label == "CO2"
    assert result.bl_label == "BL1"
    assert len(client.messages) == 2
    assert len(database.list_question_history()) == 1
    database.close()


def test_vlsi_keyword_does_not_make_an_unrelated_personal_question_valid(
    tmp_path,
):
    database = DatabaseManager(tmp_path / "vlsi.sqlite3")
    CourseManager(database).add_course(
        "VLSI Design",
        "EC3CO20",
        {
            "CO1": "Explain VLSI design principles and integrated circuits.",
            "CO2": "Analyze CMOS fabrication and semiconductor devices.",
        },
    )
    syllabus = reviewed_syllabus("EC3CO20")
    syllabus["course_name"] = "VLSI Design"
    syllabus["units"] = [
        {
            "unit_number": 1,
            "unit_title": "VLSI design",
            "topics": ["Integrated circuits", "CMOS fabrication"],
        }
    ]
    database.save_syllabus_version(syllabus, tmp_path / "vlsi.json")
    client = FakeChatClient(["CO1", "BL1"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    with pytest.raises(
        PredictionInputError,
        match="valid academic question related to the selected course syllabus",
    ):
        predictor.predict("EC3CO20", "what is yash vlsi khandel")

    assert client.messages == []
    assert database.list_question_history() == []
    database.close()


def test_short_vlsi_question_with_relevant_course_terms_is_accepted(tmp_path):
    database = DatabaseManager(tmp_path / "vlsi.sqlite3")
    CourseManager(database).add_course(
        "VLSI Design",
        "EC3CO20",
        {"CO1": "Explain VLSI design principles and integrated circuits."},
    )
    client = FakeChatClient(["CO1", "BL1"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    result = predictor.predict("EC3CO20", "Explain VLSI design.")

    assert result.co_label == "CO1"
    assert len(client.messages) == 2
    database.close()


def test_short_vlsi_definition_question_is_not_blocked_by_identity_guard(
    tmp_path,
):
    database = DatabaseManager(tmp_path / "vlsi.sqlite3")
    CourseManager(database).add_course(
        "VLSI Design",
        "EC3CO20",
        {"CO1": "Explain VLSI design principles and integrated circuits."},
    )
    client = FakeChatClient(["CO1", "BL1"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    result = predictor.predict("EC3CO20", "What is VLSI?")

    assert result.co_label == "CO1"
    assert len(client.messages) == 2
    database.close()


def test_course_without_outcomes_stops_before_inference(tmp_path, monkeypatch):
    database = make_database(tmp_path)
    course = database.get_course("CS305")
    assert course is not None
    incomplete_course = dict(course)
    incomplete_course["course_outcomes"] = dict(course["course_outcomes"])
    incomplete_course["course_outcomes"].clear()
    monkeypatch.setattr(
        database, "get_course", lambda _course_code: incomplete_course
    )
    client = FakeChatClient(["CO1", "BL1"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    with pytest.raises(PredictionInputError, match="at least one valid CO1–CO5"):
        predictor.predict("CS305", "Explain how routers forward packets.")

    assert client.messages == []
    assert database.list_question_history() == []
    database.close()


def test_prediction_accepts_only_saved_subset_including_co5(tmp_path):
    database = DatabaseManager(tmp_path / "predictor.sqlite3")
    CourseManager(database).add_course(
        "Computer Networks",
        "CS305",
        {
            "CO1": "Explain network routing models.",
            "CO5": "Evaluate network routing security.",
        },
    )
    client = FakeChatClient(["CO5", "BL4"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    result = predictor.predict("CS305", "Evaluate network routing security.")

    assert result.co_label == "CO5"
    co_prompt = client.messages[0][0][1]["content"]
    assert "CO5" in co_prompt
    assert "CO2" not in co_prompt
    assert "CO4" not in co_prompt
    database.close()


def test_long_unicode_question_is_preserved_and_formatted(tmp_path):
    database = make_database(tmp_path)
    question_text = (
        "分析网络路由：用 Dijkstra 算法计算最短路径。"
        "📡 Δ-router → node β. " * 1200
    )
    client = FakeChatClient(["CO2", "BL3"])
    predictor = QuestionPredictor(database, client, load_bloom_config())

    result = predictor.predict("CS305", question_text)

    assert result.format_output() == "CO level-CO2\nBL level - BL3"
    assert database.list_question_history()[0]["question_text"] == question_text.strip()
    assert question_text.strip() in client.messages[0][0][1]["content"]
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
