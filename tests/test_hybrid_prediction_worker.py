import json
from pathlib import Path

from src.courses.course_manager import CourseManager
from src.database.db_manager import DatabaseManager
from src.prediction import hybrid_prediction_worker as worker_module
from src.prediction.hybrid_prediction_worker import HybridPredictionWorker
from src.research.academic_evidence_analyzer import (
    AcademicEvidenceReport,
    CandidateCOAnalysis,
    COEvidence,
    EvidenceSourceReference,
)
from src.research.research_engine import ResearchEngineResult, ResearchSource
from src.research.syllabus_query_generator import SyllabusAwareQueryGenerator


OUTCOMES = {
    "CO1": "Analyze network routing algorithms and shortest paths.",
    "CO2": "Apply data link framing and error detection.",
}
UNITS = [
    {
        "unit_number": 1,
        "unit_title": "Routing",
        "topics": ["Dijkstra algorithm", "Shortest paths"],
    },
    {
        "unit_number": 2,
        "unit_title": "Data link",
        "topics": ["Framing", "Error detection"],
    },
]
QUESTION = "Analyze Dijkstra routing algorithms for shortest paths."


class FakeLLM:
    model_path = Path("test-local-model.gguf")

    def __init__(self):
        self.responses = [
            json.dumps({"label": "CO1"}),
            json.dumps({"label": "BL4"}),
        ]

    def chat(self, _messages, **_kwargs):
        return self.responses.pop(0)


def _database(tmp_path):
    database = DatabaseManager(tmp_path / "hybrid-worker.sqlite3")
    CourseManager(database).add_course("Computer Networks", "CS305", OUTCOMES)
    course = database.get_course("CS305")
    assert course is not None
    pdf_path = tmp_path / "network-syllabus.pdf"
    pdf_path.write_bytes(b"test syllabus asset")
    database.save_syllabus_version(
        {
            "course_name": "Computer Networks",
            "course_code": "CS305",
            "source_file": str(pdf_path),
            "review_status": "reviewed",
            "course_outcomes": [],
            "units": UNITS,
        },
        tmp_path / "network-syllabus.json",
    )
    return database, course


def test_local_only_worker_preserves_prediction_history_and_reports_mode(
    monkeypatch, tmp_path
):
    database, course = _database(tmp_path)
    llm_client = FakeLLM()
    monkeypatch.setattr(worker_module, "LLMClient", lambda: llm_client)
    worker = HybridPredictionWorker(
        database,
        course["id"],
        QUESTION,
        web_research_enabled=False,
    )
    completed = []
    worker.completed.connect(lambda output, status: completed.append((output, status)))

    worker.run()

    assert completed == [
        (
            "CO level - CO1\nBL level - BL4",
            "Local-only suggestion ready; optional public research is off.",
        )
    ]
    history = database.list_question_history()
    assert len(history) == 1
    assert history[0]["predicted_co"] == "CO1"
    assert history[0]["predicted_bl"] == "BL4"
    assert history[0]["predicted_bl_scheme"] == "revised_bl1_bl6"
    assert history[0]["pipeline_mode"] == "local-only"
    assert json.loads(history[0]["decision_record_json"])["selected_co"] == "CO1"
    assert len(llm_client.responses) == 0
    database.close()


def test_web_enabled_worker_runs_research_and_stores_source_metadata(
    monkeypatch, tmp_path
):
    database, course = _database(tmp_path)
    llm_client = FakeLLM()
    monkeypatch.setattr(worker_module, "LLMClient", lambda: llm_client)

    class FakeResearchEngine:
        def __init__(self, **options):
            assert options["max_queries"] <= 3
            assert options["max_pages_to_extract"] <= 2

        def research(self, **inputs):
            query_analysis = SyllabusAwareQueryGenerator().generate(
                academic_question=inputs["academic_question"],
                course_name=inputs["course_name"],
                course_code=inputs["course_code"],
                course_outcomes=inputs["course_outcomes"],
                unit_topics=inputs["unit_topics"],
                relevant_technical_concepts=inputs[
                    "relevant_technical_concepts"
                ],
                detected_cognitive_action=inputs[
                    "detected_cognitive_action"
                ],
            )
            co_id = query_analysis.plausible_co_candidates[0]
            source = ResearchSource(
                candidate_co_id=co_id,
                search_query="Dijkstra routing shortest paths",
                title="Routing Reference",
                url="https://example.edu/routing",
                search_snippet="Dijkstra finds shortest paths.",
                extracted_content="A bounded extracted source passage.",
                publication_date="2025-01-01",
                retrieved_at="2026-11-02T00:00:00+00:00",
                search_provider="DDGS",
                extraction_status="success",
                relevance_score=0.9,
                source_quality="university",
            )
            self.result = ResearchEngineResult(
                status="success",
                query_analysis=query_analysis,
                candidates=(),
                sources=(source,),
                search_query_count=1,
                retrieved_at=source.retrieved_at,
            )
            return self.result

    class FakeAcademicAnalyzer:
        def __init__(self, **_options):
            pass

        def analyze(self, **inputs):
            source = inputs["ddgs_search_results"][0]
            candidates = inputs[
                "nlp_candidate_co_rankings"
            ].plausible_co_candidates
            analyses = tuple(
                CandidateCOAnalysis(
                    co_id=co_id,
                    description=inputs["official_course_outcomes"][co_id],
                    question_objective="Analyze a routing algorithm.",
                    syllabus_relationship="Matches the reviewed routing unit.",
                    technical_concept_relevance="Dijkstra is a routing algorithm.",
                    learning_outcome_relevance="The question matches this CO.",
                    analysis="The supplied evidence supports this candidate.",
                )
                for co_id in candidates
            )
            ref = EvidenceSourceReference(
                source_id="SRC-0001",
                candidate_co_id=source.candidate_co_id,
                title=source.title,
                url=source.url,
                search_snippet=source.search_snippet,
                extracted_content=source.extracted_content,
                publication_date=source.publication_date,
                retrieved_at=source.retrieved_at,
                search_provider=source.search_provider,
                extraction_status=source.extraction_status,
                source_quality=source.source_quality,
            )
            return AcademicEvidenceReport(
                extracted_concepts=("Dijkstra", "routing", "shortest paths"),
                cognitive_action="Analyze",
                candidate_co_analysis=analyses,
                co_evidence=(
                    COEvidence(
                        co_id=source.candidate_co_id,
                        source_id="SRC-0001",
                        relationship="supports",
                        rationale="The source describes shortest-path routing.",
                    ),
                ),
                possible_bl_levels=("BL6",),
                bl_reasoning="The task requires creating a new solution.",
                bl_source_ids=("SRC-0001",),
                actual_source_ids=("SRC-0001",),
                source_catalog=(ref,),
                limitations=(),
                co_uncertain=False,
                bl_uncertain=False,
            )

    monkeypatch.setattr(worker_module, "ResearchEngine", FakeResearchEngine)
    monkeypatch.setattr(
        worker_module, "AcademicEvidenceAnalyzer", FakeAcademicAnalyzer
    )
    worker = HybridPredictionWorker(
        database,
        course["id"],
        QUESTION,
        web_research_enabled=True,
    )
    completed = []
    worker.completed.connect(lambda output, status: completed.append((output, status)))

    worker.run()

    assert completed[0][0] == "CO level - CO1\nBL level - BL6"
    assert completed[0][1] == "Hybrid suggestion ready for faculty review."
    history = database.list_question_history()
    assert history[0]["predicted_bl"] == "BL6"
    assert history[0]["predicted_bl_scheme"] == "revised_bl1_bl6"
    saved = json.loads(history[0]["decision_record_json"])
    assert saved["sources_used"][0]["search_provider"] == "DDGS"
    assert saved["sources_used"][0]["extracted_content"] == (
        "A bounded extracted source passage."
    )
    database.close()
