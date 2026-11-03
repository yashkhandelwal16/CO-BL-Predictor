from types import SimpleNamespace

from src.prediction.web_search_prediction import WebSearchPredictionService
from src.prediction.web_search_worker import WebSearchPredictionWorker
from src.research.research_engine import ResearchEngine, ResearchSource
from src.research.syllabus_query_generator import SyllabusAwareQueryGenerator
from src.research.web_search import SearchUnavailableError, WebSearchService


OUTCOMES = {
    "CO1": "Analyze network routing algorithms and shortest paths using Dijkstra.",
    "CO2": "Apply data link error detection and framing.",
}
UNITS = [
    {
        "unit_title": "Network routing",
        "topics": ["Dijkstra algorithm", "Shortest paths"],
    },
    {
        "unit_title": "Data link",
        "topics": ["Error detection", "Framing"],
    },
]
QUESTION = "Analyze Dijkstra routing and shortest paths."


class FakeDDGS:
    def __init__(self, *, timeout):
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def text(self, query, **_options):
        assert "dijkstra" in query.casefold()
        return [
            {
                "title": "Dijkstra shortest-path routing",
                "href": "https://docs.example.edu/routing",
                "body": "Dijkstra computes shortest paths in network routing.",
            }
        ]


class FakeExtractionService:
    def clear_cache(self):
        return None

    def extract_results(self, _results, *, cancellation_event=None):
        return []


class FailingSearchService:
    def search(self, _query, *, max_results):
        raise SearchUnavailableError("Web search is unavailable.")


def _source(co_id, score):
    return ResearchSource(
        candidate_co_id=co_id,
        search_query="Dijkstra routing",
        title="Dijkstra routing documentation",
        url=f"https://docs.example.edu/{co_id.casefold()}",
        search_snippet="Dijkstra computes shortest paths.",
        extracted_content=None,
        publication_date=None,
        retrieved_at="2026-11-02T00:00:00+00:00",
        search_provider="DDGS",
        extraction_status="skipped",
        relevance_score=score,
        source_quality="official_documentation",
    )


class StaticResearchEngine:
    def __init__(self, status, sources=(), message=None, query_analysis=None):
        self.result = SimpleNamespace(
            status=status,
            sources=tuple(sources),
            message=message,
            query_analysis=query_analysis or SimpleNamespace(co_rankings=()),
        )
        self.inputs = None

    def research(self, **inputs):
        self.inputs = inputs
        return self.result


def test_ddgs_pipeline_needs_no_api_key_and_uses_course_context():
    service = WebSearchService(ddgs_factory=FakeDDGS)
    engine = ResearchEngine(
        search_service=service,
        extraction_service=FakeExtractionService(),
        max_queries=2,
        max_pages_to_extract=1,
    )

    result = WebSearchPredictionService(engine).predict(
        question=QUESTION,
        course_name="Computer Networks",
        course_code="CS305",
        course_outcomes=OUTCOMES,
        syllabus_units=UNITS,
    )

    assert result.format_output() == "CO level - CO1\nBL level - BL4"
    assert result.status_message == "Web Search suggestion ready for faculty review."


def test_web_co_and_bl_uncertainty_are_independent():
    engine = StaticResearchEngine(
        "success",
        sources=(_source("CO1", 0.8), _source("CO2", 0.8)),
    )
    result = WebSearchPredictionService(engine).predict(
        question=QUESTION,
        course_name="Computer Networks",
        course_code="CS305",
        course_outcomes=OUTCOMES,
        syllabus_units=UNITS,
    )

    assert result.co_label == "N/A"
    assert result.bl_label == "BL4"
    assert engine.inputs["course_outcomes"] == OUTCOMES
    assert engine.inputs["unit_topics"] == UNITS


def test_web_co_uses_clear_syllabus_ranking_when_search_has_no_match():
    query_analysis = SyllabusAwareQueryGenerator().generate(
        academic_question=QUESTION,
        course_name="Computer Networks",
        course_code="CS305",
        course_outcomes=OUTCOMES,
        unit_topics=UNITS,
        relevant_technical_concepts=(),
        detected_cognitive_action="Analyze",
    )
    engine = StaticResearchEngine(
        "success",
        query_analysis=query_analysis,
    )

    result = WebSearchPredictionService(engine).predict(
        question=QUESTION,
        course_name="Computer Networks",
        course_code="CS305",
        course_outcomes=OUTCOMES,
        syllabus_units=UNITS,
    )

    assert result.co_label == "CO1"
    assert "best course-context CO match" in result.status_message


def test_web_co_shows_best_nonzero_course_match_below_search_threshold():
    query_analysis = SimpleNamespace(
        co_rankings=(
            SimpleNamespace(co_id="CO1", similarity=0.005),
            SimpleNamespace(co_id="CO2", similarity=0.003),
        ),
        relevant_syllabus_topics=(
            SimpleNamespace(unit="OOP basics", topic="Classes and Objects"),
        ),
    )
    engine = StaticResearchEngine(
        "no_candidates",
        message="No course outcome had enough evidence to justify a web search.",
        query_analysis=query_analysis,
    )

    result = WebSearchPredictionService(engine).predict(
        question="Explain with example what is classes?",
        course_name="Object Oriented Programming using Java",
        course_code="CS3013",
        course_outcomes=OUTCOMES,
        syllabus_units=UNITS,
    )

    assert result.co_label == "CO1"
    assert "best course-context CO match" in result.status_message
    assert "No web query was run" in result.status_message


def test_web_co_is_not_na_for_screenshot_style_singular_plural_question():
    outcomes = {
        "CO1": "Remember OOP principles and Java programming basics.",
        "CO2": "Understand inheritance and polymorphism concepts.",
        "CO3": "Apply OOP concepts to develop programs in Java.",
        "CO4": "Analyze class relationships and object interactions.",
    }
    units = [
        {
            "unit_title": "Classes and Objects",
            "topics": ["Classes", "Objects", "Methods", "Encapsulation"],
        }
    ]
    question = "illustrate is classes and object ?"
    query_analysis = SyllabusAwareQueryGenerator().generate(
        academic_question=question,
        course_name="Object Oriented Programming using Java",
        course_code="CS3013",
        course_outcomes=outcomes,
        unit_topics=units,
        relevant_technical_concepts=(),
        detected_cognitive_action="illustrate",
    )
    service = WebSearchPredictionService(
        StaticResearchEngine("success", query_analysis=query_analysis)
    )

    result = service.predict(
        question=question,
        course_name="Object Oriented Programming using Java",
        course_code="CS3013",
        course_outcomes=outcomes,
        syllabus_units=units,
    )

    assert result.co_label in outcomes
    assert result.co_label != "N/A"


def test_web_search_uses_six_level_taxonomy_independently_of_search_results():
    engine = StaticResearchEngine("success")
    service = WebSearchPredictionService(engine)

    expected_levels = (
        ("Define inheritance.", "BL1"),
        ("Explain polymorphism.", "BL2"),
        ("Implement a stack using an array.", "BL3"),
        ("Analyze the relationship between two classes.", "BL4"),
        ("Evaluate two algorithms using performance criteria.", "BL5"),
        ("Design a new network routing algorithm.", "BL6"),
        ("A long, difficult programming question.", "N/A"),
    )
    for question, expected in expected_levels:
        result = service.predict(
            question=question,
            course_name="Computer Networks",
            course_code="CS305",
            course_outcomes=OUTCOMES,
            syllabus_units=UNITS,
        )
        assert result.bl_label == expected


def test_search_failure_returns_unavailable_co_without_blocking_bl():
    engine = ResearchEngine(
        search_service=FailingSearchService(),
        extraction_service=FakeExtractionService(),
        max_queries=2,
        max_pages_to_extract=1,
    )

    result = WebSearchPredictionService(engine).predict(
        question=QUESTION,
        course_name="Computer Networks",
        course_code="CS305",
        course_outcomes=OUTCOMES,
        syllabus_units=UNITS,
    )

    assert result.co_label == "CO1"
    assert result.bl_label == "BL4"
    assert "unavailable" in result.status_message.casefold()
    assert "course-context" in result.status_message.casefold()


def test_unexpected_worker_error_is_reported_as_web_only_status():
    class BrokenService:
        def predict(self, **_kwargs):
            raise RuntimeError("private implementation detail")

    worker = WebSearchPredictionWorker(
        question=QUESTION,
        course_name="Computer Networks",
        course_code="CS305",
        course_outcomes=OUTCOMES,
        syllabus_units=UNITS,
        service=BrokenService(),
    )
    completed = []
    worker.completed.connect(
        lambda output, status: completed.append((output, status))
    )

    worker.run()

    assert completed == [
        (
            "CO level - N/A\nBL level - BL4",
            "Web search could not be completed. The existing suggestion is unaffected.",
        )
    ]
