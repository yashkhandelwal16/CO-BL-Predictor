import json
from threading import Event

import pytest

from src.research import (
    AcademicEvidenceAnalyzer,
    InvalidEvidenceReferenceError,
    MalformedEvidenceResponseError,
    ResearchSource,
)
from src.research.syllabus_query_generator import (
    CourseOutcomeRanking,
    SyllabusTopicMatch,
)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.response


def _valid_response():
    return {
        "extracted_concepts": ["inheritance", "polymorphism"],
        "cognitive_action": "Implement",
        "candidate_co_analysis": [
            {
                "co_id": "CO2",
                "question_objective": "Apply inheritance to a programming task.",
                "syllabus_relationship": "Matches the reviewed OOP unit.",
                "technical_concept_relevance": "Inheritance is directly relevant.",
                "learning_outcome_relevance": "The task exercises the stated outcome.",
                "analysis": "The available course context supports CO2.",
            }
        ],
        "co_evidence": [
            {
                "co_id": "CO2",
                "source_id": "SRC-0001",
                "relationship": "supports",
                "rationale": "The source describes Java inheritance.",
            }
        ],
        "possible_bl_levels": ["BL3"],
        "bl_reasoning": "The required task is to apply a known programming concept.",
        "bl_source_ids": ["SRC-0001"],
        "limitations": ["The web source is supporting context, not institutional authority."],
        "co_uncertain": False,
        "bl_uncertain": False,
    }


def _context():
    ranking = CourseOutcomeRanking(
        co_id="CO2",
        description="Implement inheritance and polymorphism in Java.",
        similarity=0.82,
        tfidf_similarity=0.76,
        concept_overlap=1.0,
        topic_similarity=0.6,
        matched_topics=(
            SyllabusTopicMatch("Unit II", "Inheritance", 0.6),
        ),
    )
    source = ResearchSource(
        candidate_co_id="CO2",
        search_query="Java inheritance concepts",
        title="Java Inheritance Tutorial",
        url="https://docs.oracle.com/java/tutorial/inheritance.html",
        search_snippet="Java supports inheritance between classes.",
        extracted_content="A class can inherit fields and methods from another class.",
        publication_date="2025-01-01",
        retrieved_at="2026-11-02T00:00:00+00:00",
        search_provider="DDGS",
        extraction_status="success",
        relevance_score=0.8,
        source_quality="official_documentation",
    )
    return {
        "academic_question": "Implement a Java class using inheritance.",
        "course_name": "Object Oriented Programming using Java",
        "course_code": "CS201",
        "official_course_outcomes": {
            "CO1": "Recall Java syntax and primitive data types.",
            "CO2": "Implement inheritance and polymorphism in Java.",
        },
        "relevant_syllabus_units": {"Unit II": ["Inheritance", "Polymorphism"]},
        "nlp_candidate_co_rankings": [ranking],
        "ddgs_search_results": [source],
        "faculty_approved_examples": [],
    }


def test_analyzer_validates_and_returns_evidence_with_actual_source_ids():
    llm = FakeLLM(json.dumps(_valid_response()))
    analyzer = AcademicEvidenceAnalyzer(llm_client=llm)

    report = analyzer.analyze(**_context())

    assert report.candidate_co_analysis[0].co_id == "CO2"
    assert report.candidate_co_analysis[0].description == (
        "Implement inheritance and polymorphism in Java."
    )
    assert report.possible_bl_levels == ("BL3",)
    assert report.actual_source_ids == ("SRC-0001",)
    assert report.source_catalog[0].url == (
        "https://docs.oracle.com/java/tutorial/inheritance.html"
    )
    assert json.loads(report.to_json())["actual_source_ids"] == ["SRC-0001"]
    assert llm.calls[0][1]["temperature"] == 0.0
    assert "Never answer" in llm.calls[0][0][0]["content"]


def test_analyzer_rejects_malformed_json_without_repairing_it():
    analyzer = AcademicEvidenceAnalyzer(llm_client=FakeLLM("```json\n{}\n```"))

    with pytest.raises(MalformedEvidenceResponseError):
        analyzer.analyze(**_context())


def test_analyzer_allows_missing_web_evidence_and_keeps_bl_uncertainty_separate():
    response = _valid_response()
    response["co_evidence"] = []
    response["possible_bl_levels"] = ["N/A"]
    response["bl_source_ids"] = []
    response["bl_reasoning"] = "No evidence was available to distinguish a level."
    response["co_uncertain"] = False
    response["bl_uncertain"] = True
    context = _context()
    context["ddgs_search_results"] = []
    analyzer = AcademicEvidenceAnalyzer(llm_client=FakeLLM(json.dumps(response)))

    report = analyzer.analyze(**context)

    assert report.actual_source_ids == ()
    assert report.co_uncertain is False
    assert report.bl_uncertain is True
    assert report.possible_bl_levels == ("N/A",)


def test_analyzer_rejects_forged_source_ids():
    response = _valid_response()
    response["co_evidence"][0]["source_id"] = "SRC-9999"
    analyzer = AcademicEvidenceAnalyzer(llm_client=FakeLLM(json.dumps(response)))

    with pytest.raises(InvalidEvidenceReferenceError, match="not retrieved"):
        analyzer.analyze(**_context())


def test_analyzer_rejects_non_candidate_or_non_official_co_ids():
    response = _valid_response()
    response["candidate_co_analysis"][0]["co_id"] = "CO1"
    analyzer = AcademicEvidenceAnalyzer(llm_client=FakeLLM(json.dumps(response)))

    with pytest.raises(InvalidEvidenceReferenceError):
        analyzer.analyze(**_context())


def test_analyzer_passes_cancellation_event_to_local_model():
    cancellation_event = Event()
    llm = FakeLLM(json.dumps(_valid_response()))
    analyzer = AcademicEvidenceAnalyzer(llm_client=llm)

    analyzer.analyze(**_context(), cancellation_event=cancellation_event)

    assert llm.calls[0][1]["cancellation_event"] is cancellation_event
