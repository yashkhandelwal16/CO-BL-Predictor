from types import SimpleNamespace

from src.prediction.web_search_prediction import WebSearchPredictionService
from src.prediction.web_search_worker import WebSearchPredictionWorker
from src.research.research_engine import ResearchEngine, ResearchSource
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
    instances = []

    def __init__(self, *, timeout):
        self.timeout = timeout
        self.queries = []
        self.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def text(self, query, **_options):
        self.queries.append(query)
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
    def __init__(self, status, sources=(), message=None):
        self.result = SimpleNamespace(
            status=status,
            sources=tuple(sources),
            message=message,
        )
        self.inputs = None

    def research(self, **inputs):
        self.inputs = inputs
        return self.result


def test_ddgs_pipeline_needs_no_api_key_and_uses_course_context():
    FakeDDGS.instances.clear()
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
    query = FakeDDGS.instances[0].queries[0]
    assert "dijkstra" in query.casefold()
    assert QUESTION.casefold() not in query.casefold()
    assert "cs305" not in query.casefold()
    assert OUTCOMES["CO1"].casefold() not in query.casefold()


def test_ambiguous_web_evidence_uses_best_syllabus_co_and_keeps_bl_independent():
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

    assert result.co_label == "CO1"
    assert result.bl_label == "BL4"
    assert "syllabus-based" in result.status_message
    assert engine.inputs["course_outcomes"] == OUTCOMES
    assert engine.inputs["unit_topics"] == UNITS


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
    assert "syllabus-based" in result.status_message.casefold()


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
