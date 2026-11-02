from threading import Event

import pytest

from src.research import (
    ResearchEngine,
    ResearchWorker,
    SearchResult,
    SyllabusAwareQueryGenerator,
    WebContentResult,
    WebSearchError,
)


class FakeSearchService:
    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search(self, query, *, max_results):
        self.calls.append((query, max_results))
        if self.error is not None:
            raise self.error
        return list(self.results[:max_results])


class FakeExtractionService:
    def __init__(self, status="success", error=None):
        self.status = status
        self.error = error
        self.calls = []
        self.clear_count = 0

    def clear_cache(self):
        self.clear_count += 1

    def extract_results(self, results, *, cancellation_event=None):
        if cancellation_event is not None and cancellation_event.is_set():
            return []
        self.calls.append(tuple(results))
        if self.error is not None:
            raise self.error
        return [
            WebContentResult(
                title=result.title,
                url=result.url,
                snippet=result.snippet,
                content=(
                    "Extracted technical explanation."
                    if self.status in {"success", "cached"}
                    else result.snippet
                ),
                source=result.source,
                metadata=result.metadata,
                publication_date="2026-01-15",
                retrieved_at="2026-11-02T00:00:00+00:00",
                extraction_status=self.status,
            )
            for result in results
        ]


def _context(
    question="Explain Java inheritance and polymorphism.",
    concepts=("Java", "inheritance", "polymorphism"),
    action="Understand",
    outcomes=None,
    topics=None,
):
    return {
        "academic_question": question,
        "course_name": "Object Oriented Programming using Java",
        "course_code": "CS201",
        "course_outcomes": outcomes
        or {
            "CO1": "Recall Java syntax and primitive data types.",
            "CO2": "Implement inheritance and polymorphism in Java.",
            "CO3": "Analyze exception handling and file operations.",
        },
        "unit_topics": topics or {
            "Unit II: Object Oriented Programming": [
                "Inheritance and polymorphism",
                "Method overriding",
            ],
            "Unit III: Robust Programs": ["Exception handling"],
        },
        "relevant_technical_concepts": concepts,
        "detected_cognitive_action": action,
    }


def _results():
    return [
        SearchResult(
            "Java inheritance tutorial",
            "https://docs.oracle.com/java/tutorial/inheritance.html",
            "Java classes use inheritance and polymorphism to reuse behavior.",
            source="Oracle",
            metadata={"date": "2025-01-01"},
        ),
        SearchResult(
            "Inheritance course notes",
            "https://cs.example.edu/inheritance",
            "Inheritance and polymorphism are core object oriented programming concepts.",
            source="University",
        ),
        SearchResult(
            "Butterfly migration",
            "https://biology.example.org/migration",
            "Monarch butterflies migrate seasonally.",
        ),
    ]


def test_research_associates_ranked_results_and_extraction_with_candidate():
    search = FakeSearchService(_results())
    extraction = FakeExtractionService()
    engine = ResearchEngine(
        search_service=search,
        extraction_service=extraction,
        max_pages_to_extract=2,
    )

    result = engine.research(**_context())

    assert result.status == "success"
    assert result.search_query_count == 1
    assert result.candidates[0].co_id == "CO2"
    assert result.candidates[0].status == "success"
    assert len(result.sources) == 2
    assert len(extraction.calls) == 1
    assert all(source.candidate_co_id == "CO2" for source in result.sources)
    assert all(source.search_query == search.calls[0][0] for source in result.sources)
    assert all(source.search_provider == "DDGS" for source in result.sources)
    assert all(source.extraction_status == "success" for source in result.sources)
    assert all(source.extracted_content == "Extracted technical explanation." for source in result.sources)
    assert all(source.retrieved_at == "2026-11-02T00:00:00+00:00" for source in result.sources)
    assert all(source.publication_date == "2026-01-15" for source in result.sources)
    assert {source.source_quality for source in result.sources} <= {
        "university",
        "official_documentation",
    }
    assert all(
        source.url != "https://biology.example.org/migration/"
        for source in result.sources
    )


def test_external_query_excludes_question_course_and_official_syllabus_text():
    question = "Explain Java inheritance and polymorphism with this private exam text."
    outcomes = {
        "CO1": "Recall private lab handbook content and Java syntax.",
        "CO2": "Implement inheritance and polymorphism in Java.",
    }
    search = FakeSearchService(_results())
    engine = ResearchEngine(
        search_service=search,
        extraction_service=FakeExtractionService(),
    )

    result = engine.research(**_context(question=question, outcomes=outcomes))

    assert result.status == "success"
    sent_query = search.calls[0][0]
    assert result.candidates[0].search_query == sent_query
    assert question not in sent_query
    assert "private exam text" not in sent_query
    assert "CS201" not in sent_query
    assert "private lab handbook content" not in sent_query
    assert "Unit II" not in sent_query
    assert "inheritance" in sent_query


def test_search_cache_prevents_duplicate_external_query_and_reuses_results():
    search = FakeSearchService(_results())
    extraction = FakeExtractionService()
    engine = ResearchEngine(
        search_service=search,
        extraction_service=extraction,
    )

    first = engine.research(**_context())
    second = engine.research(**_context())

    assert first.sources
    assert second.sources
    assert len(search.calls) == 1
    assert len(extraction.calls) == 2


def test_failed_duplicate_search_query_is_not_retried_for_another_co():
    outcomes = {
        "CO1": "Apply inheritance and polymorphism in Java programs.",
        "CO2": "Analyze inheritance and polymorphism in Java programs.",
    }
    search = FakeSearchService(error=WebSearchError("provider blocked"))
    engine = ResearchEngine(
        search_service=search,
        extraction_service=FakeExtractionService(),
        query_generator=SyllabusAwareQueryGenerator(max_candidates=2),
    )

    result = engine.research(**_context(outcomes=outcomes))

    assert len(result.candidates) == 2
    assert all(candidate.status == "search_failed" for candidate in result.candidates)
    assert len(search.calls) == 1
    assert result.search_query_count == 1
    assert result.status == "search_unavailable"


def test_empty_search_results_are_cached_and_return_successful_empty_status():
    search = FakeSearchService([])
    engine = ResearchEngine(
        search_service=search,
        extraction_service=FakeExtractionService(),
    )

    result = engine.research(**_context())

    assert result.status == "success"
    assert result.sources == ()
    assert result.candidates[0].status == "empty"
    assert result.candidates[0].result_count == 0
    assert result.message


def test_search_cache_expires_after_configured_time():
    now = [0.0]
    search = FakeSearchService(_results())
    engine = ResearchEngine(
        search_service=search,
        extraction_service=FakeExtractionService(),
        cache_seconds=10,
        clock=lambda: now[0],
    )

    engine.research(**_context())
    now[0] = 11.0
    engine.research(**_context())

    assert len(search.calls) == 2


def test_clear_cache_clears_search_and_extraction_caches():
    search = FakeSearchService(_results())
    extraction = FakeExtractionService()
    engine = ResearchEngine(search_service=search, extraction_service=extraction)

    engine.research(**_context())
    engine.clear_cache()
    engine.research(**_context())

    assert len(search.calls) == 2
    assert extraction.clear_count == 1


def test_search_failure_returns_empty_evidence_with_meaningful_status():
    search = FakeSearchService(error=WebSearchError("Search provider unavailable."))
    engine = ResearchEngine(
        search_service=search,
        extraction_service=FakeExtractionService(),
    )

    result = engine.research(**_context())

    assert result.status == "search_unavailable"
    assert result.sources == ()
    assert result.candidates[0].status == "search_failed"
    assert result.candidates[0].error == "Search provider unavailable."
    assert result.message


def test_mixed_candidate_search_failure_is_reported_as_partial():
    outcomes = {
        "CO1": "Explain network models and protocol layering.",
        "CO2": "Analyze routing algorithms and packet forwarding.",
    }
    results = [
        SearchResult(
            "Network protocol layers",
            "https://cs.example.edu/network-models",
            "Network models explain protocol layering and communication.",
        ),
        SearchResult(
            "Routing algorithms",
            "https://docs.example.org/routing",
            "Routing algorithms determine packet forwarding paths.",
        ),
    ]
    search = FakeSearchService(
        results,
        error=WebSearchError("temporary failure"),
    )

    def fail_second_query(query, *, max_results):
        search.calls.append((query, max_results))
        if len(search.calls) == 2:
            raise WebSearchError("temporary failure")
        return list(search.results[:max_results])

    search.search = fail_second_query
    engine = ResearchEngine(
        search_service=search,
        extraction_service=FakeExtractionService(),
    )

    result = engine.research(
        **_context(
            question="Explain network protocol layers and routing algorithms.",
            concepts=("network models", "routing algorithms"),
            action="Understand",
            outcomes=outcomes,
            topics={
                "Unit I: Network Fundamentals": [
                    "Network models",
                    "Protocol layering",
                ],
                "Unit III: Network Layer": [
                    "Routing algorithms",
                    "Packet forwarding",
                ],
            },
        )
    )

    assert len(search.calls) == 2
    assert result.status == "partial"
    assert "search_failed" in {candidate.status for candidate in result.candidates}
    assert "success" in {candidate.status for candidate in result.candidates}


def test_webpage_extraction_failure_preserves_search_snippet_and_status():
    search = FakeSearchService(_results())
    extraction = FakeExtractionService(error=RuntimeError("internal detail"))
    engine = ResearchEngine(search_service=search, extraction_service=extraction)

    result = engine.research(**_context())

    assert result.sources
    assert all(source.extraction_status == "failed" for source in result.sources)
    assert all(source.search_snippet for source in result.sources)
    assert all(source.extracted_content is None for source in result.sources)
    assert all(
        source.error_details == "Webpage extraction could not be completed."
        for source in result.sources
    )


def test_extraction_service_failure_falls_back_to_snippet_status():
    search = FakeSearchService(_results())
    extraction = FakeExtractionService(status="failed")
    engine = ResearchEngine(search_service=search, extraction_service=extraction)

    result = engine.research(**_context())

    assert result.sources
    assert all(source.extraction_status == "failed" for source in result.sources)
    assert all(source.search_snippet for source in result.sources)


def test_deduplicates_result_urls_but_keeps_distinct_co_associations():
    outcomes = {
        "CO1": "Apply inheritance using Java class hierarchies.",
        "CO2": "Analyze inheritance and polymorphism in Java.",
    }
    duplicate_results = [
        SearchResult(
            "Java inheritance",
            "https://docs.oracle.com/java/tutorial/inheritance.html#one",
            "Java inheritance and polymorphism in class hierarchies.",
        ),
        SearchResult(
            "Java inheritance duplicate",
            "https://docs.oracle.com/java/tutorial/inheritance.html#two",
            "Java inheritance and polymorphism in class hierarchies.",
        ),
    ]
    search = FakeSearchService(duplicate_results)
    extraction = FakeExtractionService()
    engine = ResearchEngine(
        search_service=search,
        extraction_service=extraction,
        max_queries=2,
    )

    result = engine.research(
        **_context(
            question="Analyze Java inheritance and polymorphism in class hierarchies.",
            concepts=("Java", "inheritance", "polymorphism", "class hierarchies"),
            outcomes=outcomes,
        )
    )

    assert len(extraction.calls) == 1
    assert len(extraction.calls[0]) == 1
    assert len({source.url for source in result.sources}) == 1
    assert len(result.sources) >= 1
    assert {source.candidate_co_id for source in result.sources} <= set(outcomes)


def test_query_and_extraction_limits_are_enforced():
    outcomes = {
        "CO1": "Apply graph traversal algorithms such as BFS and DFS.",
        "CO2": "Analyze graph traversal complexity and shortest paths.",
        "CO3": "Design graph data structures and network representations.",
    }
    search = FakeSearchService(_results())
    extraction = FakeExtractionService()
    engine = ResearchEngine(
        search_service=search,
        extraction_service=extraction,
        max_queries=2,
        results_per_query=2,
        max_pages_to_extract=1,
    )

    result = engine.research(
        **_context(
            question="Analyze graph traversal with BFS and DFS algorithms.",
            concepts=("graph traversal", "BFS", "DFS"),
            outcomes=outcomes,
        )
    )

    assert len(search.calls) <= 2
    assert all(limit == 2 for _, limit in search.calls)
    assert len(extraction.calls[0]) <= 1
    assert result.search_query_count == len(search.calls)


def test_empty_plausible_candidate_set_does_not_make_search_calls():
    search = FakeSearchService(_results())
    engine = ResearchEngine(
        search_service=search,
        extraction_service=FakeExtractionService(),
    )

    result = engine.research(
        **_context(
            question="Describe monarch butterfly migration patterns.",
            concepts=(),
        )
    )

    assert result.status == "no_candidates"
    assert result.sources == ()
    assert result.search_query_count == 0
    assert search.calls == []


def test_missing_public_concepts_prevents_external_query():
    search = FakeSearchService(_results())
    engine = ResearchEngine(
        search_service=search,
        extraction_service=FakeExtractionService(),
    )

    result = engine.research(**_context(concepts=()))

    assert result.status == "no_public_queries"
    assert result.sources == ()
    assert result.search_query_count == 0
    assert result.message
    assert search.calls == []


def test_cancel_before_search_returns_cancelled_without_network_calls():
    search = FakeSearchService(_results())
    engine = ResearchEngine(
        search_service=search,
        extraction_service=FakeExtractionService(),
    )
    cancellation = Event()
    cancellation.set()

    result = engine.research(**_context(), cancellation_event=cancellation)

    assert result.status == "cancelled"
    assert result.search_query_count == 0
    assert search.calls == []
    assert all(
        _context()["academic_question"] not in candidate.search_query
        for candidate in result.candidates
    )


def test_worker_runs_engine_and_emits_structured_result():
    search = FakeSearchService(_results())
    engine = ResearchEngine(
        search_service=search,
        extraction_service=FakeExtractionService(),
    )
    worker = ResearchWorker(engine, **_context())
    emitted = []
    worker.completed.connect(emitted.append)

    worker.run()

    assert emitted
    assert emitted[0].status == "success"
    assert emitted[0].sources


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_queries", 0),
        ("results_per_query", 21),
        ("max_sources_per_co", 0),
        ("max_pages_to_extract", 11),
        ("cache_seconds", 0),
        ("max_cache_entries", 0),
        ("relevance_threshold", float("nan")),
    ],
)
def test_engine_configuration_is_validated(name, value):
    with pytest.raises(ValueError):
        ResearchEngine(**{name: value})
