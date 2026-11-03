import logging
import math
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event, RLock
from typing import Literal
from urllib.parse import urlsplit

from src.nlp.text_preprocessor import preprocess_text, tfidf_similarity_scores
from src.research.syllabus_query_generator import (
    GeneratedSearchQuery,
    SyllabusAwareQueryGenerator,
    SyllabusAwareQueryResult,
)
from src.research.url_security import InvalidResultURLError, normalize_http_url
from src.research.web_content_extraction import (
    WebContentExtractionService,
    WebContentResult,
)
from src.research.web_search import (
    SearchResponseError,
    SearchResult,
    WebSearchError,
    WebSearchService,
)


DEFAULT_MAX_QUERIES = 5
MAX_QUERIES_LIMIT = 20
DEFAULT_RESULTS_PER_QUERY = 5
MAX_RESULTS_PER_QUERY_LIMIT = 20
DEFAULT_MAX_SOURCES_PER_CO = 5
MAX_SOURCES_PER_CO_LIMIT = 20
DEFAULT_MAX_PAGES_TO_EXTRACT = 3
MAX_PAGES_TO_EXTRACT_LIMIT = 10
DEFAULT_CACHE_SECONDS = 900
DEFAULT_CACHE_ENTRIES = 256
DEFAULT_RELEVANCE_THRESHOLD = 0.025
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_EDUCATIONAL_DOMAINS = (
    "khanacademy.org",
    "openstax.org",
    "ocw.mit.edu",
    "arxiv.org",
    "doi.org",
    "ieeexplore.ieee.org",
    "acm.org",
)
_DOCUMENTATION_DOMAINS = (
    "docs.python.org",
    "developer.mozilla.org",
    "docs.oracle.com",
    "learn.microsoft.com",
    "developer.apple.com",
    "developer.android.com",
)
_LOGGER = logging.getLogger(__name__)


ResearchStatus = Literal[
    "success",
    "partial",
    "search_unavailable",
    "no_candidates",
    "no_public_queries",
    "cancelled",
]
CandidateResearchStatus = Literal[
    "success", "empty", "search_failed", "cancelled"
]
ExtractionStatus = Literal[
    "success", "cached", "failed", "skipped", "cancelled", "not_attempted"
]
SourceQuality = Literal[
    "university", "official_documentation", "academic", "educational", "general"
]


@dataclass(frozen=True)
class ResearchSource:
    candidate_co_id: str
    search_query: str
    title: str
    url: str
    search_snippet: str
    extracted_content: str | None
    publication_date: str | None
    retrieved_at: str
    search_provider: str
    extraction_status: ExtractionStatus
    relevance_score: float
    source_quality: SourceQuality
    error_details: str | None = None


@dataclass(frozen=True)
class CandidateResearch:
    co_id: str
    co_description: str
    search_query: str
    status: CandidateResearchStatus
    result_count: int
    sources: tuple[ResearchSource, ...]
    error: str | None = None


@dataclass(frozen=True)
class ResearchEngineResult:
    status: ResearchStatus
    query_analysis: SyllabusAwareQueryResult
    candidates: tuple[CandidateResearch, ...]
    sources: tuple[ResearchSource, ...]
    search_query_count: int
    retrieved_at: str
    message: str | None = None


@dataclass(frozen=True)
class _ScoredResult:
    candidate_co_id: str
    query: GeneratedSearchQuery
    search_result: SearchResult
    relevance_score: float
    source_quality: SourceQuality


@dataclass(frozen=True)
class _SearchCacheEntry:
    results: tuple[SearchResult, ...]
    expires_at: float


class ResearchEngine:
    """Orchestrate CO-targeted search and extraction without assigning labels."""

    def __init__(
        self,
        *,
        search_service: WebSearchService | None = None,
        extraction_service: WebContentExtractionService | None = None,
        query_generator: SyllabusAwareQueryGenerator | None = None,
        max_queries: int = DEFAULT_MAX_QUERIES,
        results_per_query: int = DEFAULT_RESULTS_PER_QUERY,
        max_sources_per_co: int = DEFAULT_MAX_SOURCES_PER_CO,
        max_pages_to_extract: int = DEFAULT_MAX_PAGES_TO_EXTRACT,
        cache_seconds: int = DEFAULT_CACHE_SECONDS,
        max_cache_entries: int = DEFAULT_CACHE_ENTRIES,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.search_service = search_service or WebSearchService()
        self.extraction_service = (
            extraction_service or WebContentExtractionService()
        )
        self.query_generator = query_generator or SyllabusAwareQueryGenerator()
        self.max_queries = self._validate_limit(
            max_queries, MAX_QUERIES_LIMIT, "Maximum searches"
        )
        self.results_per_query = self._validate_limit(
            results_per_query,
            MAX_RESULTS_PER_QUERY_LIMIT,
            "Results per search",
        )
        self.max_sources_per_co = self._validate_limit(
            max_sources_per_co,
            MAX_SOURCES_PER_CO_LIMIT,
            "Maximum sources per CO",
        )
        self.max_pages_to_extract = self._validate_limit(
            max_pages_to_extract,
            MAX_PAGES_TO_EXTRACT_LIMIT,
            "Maximum pages to extract",
        )
        self.cache_seconds = self._validate_positive_int(
            cache_seconds, "Search cache duration"
        )
        self.max_cache_entries = self._validate_positive_int(
            max_cache_entries, "Search cache size"
        )
        self.relevance_threshold = self._validate_threshold(
            relevance_threshold
        )
        self._clock = clock or time.monotonic
        self._search_cache: OrderedDict[str, _SearchCacheEntry] = OrderedDict()
        self._cache_lock = RLock()

    def research(
        self,
        academic_question: str,
        course_name: str,
        course_code: str,
        course_outcomes: Mapping[str, str],
        unit_topics: (
            Mapping[str, Sequence[str]]
            | Sequence[Mapping[str, object]]
        ),
        relevant_technical_concepts: Sequence[str],
        detected_cognitive_action: str,
        *,
        cancellation_event: Event | None = None,
    ) -> ResearchEngineResult:
        """Search and extract evidence independently for plausible COs."""
        analysis = self.query_generator.generate(
            academic_question=academic_question,
            course_name=course_name,
            course_code=course_code,
            course_outcomes=course_outcomes,
            unit_topics=unit_topics,
            relevant_technical_concepts=relevant_technical_concepts,
            detected_cognitive_action=detected_cognitive_action,
        )
        outcomes = {
            ranking.co_id: ranking.description
            for ranking in analysis.co_rankings
        }
        candidate_queries = analysis.generated_queries[: self.max_queries]
        if not candidate_queries:
            return self._result(
                "no_candidates",
                analysis,
                (),
                (),
                0,
                "No course outcome had enough evidence to justify a web search.",
            )

        collected: list[_ScoredResult] = []
        candidates: list[CandidateResearch] = []
        queries_attempted = 0
        query_attempts: dict[str, tuple[tuple[SearchResult, ...] | None, str | None]] = {}
        for generated_query in candidate_queries:
            candidate_ranking = next(
                (
                    ranking
                    for ranking in analysis.co_rankings
                    if ranking.co_id == generated_query.co_id
                ),
                None,
            )
            candidate_topics = (
                tuple(topic.topic for topic in candidate_ranking.matched_topics)
                if candidate_ranking is not None
                else ()
            )
            safe_query = self._privacy_safe_query(
                relevant_technical_concepts,
                detected_cognitive_action,
                outcomes[generated_query.co_id],
                candidate_topics,
            )
            if not safe_query:
                candidates.append(
                    CandidateResearch(
                        co_id=generated_query.co_id,
                        co_description=outcomes[generated_query.co_id],
                        search_query="",
                        status="empty",
                        result_count=0,
                        sources=(),
                        error=(
                            "No privacy-safe technical concepts were available "
                            "to create a public search query."
                        ),
                    )
                )
                continue
            safe_generated_query = GeneratedSearchQuery(
                co_id=generated_query.co_id,
                query=safe_query,
                reason=(
                    generated_query.reason
                    + " The external query excludes the question, course "
                    "identity, official outcome text, and syllabus topics."
                ),
            )
            if self._is_cancelled(cancellation_event):
                candidates.append(
                    self._candidate_cancelled(
                        safe_generated_query, outcomes[generated_query.co_id]
                    )
                )
                continue
            query_error = None
            query_key = preprocess_text(safe_query)
            if query_key not in query_attempts:
                queries_attempted += 1
                try:
                    query_attempts[query_key] = (self._search(safe_query), None)
                except Exception as error:
                    if isinstance(error, WebSearchError):
                        safe_message = str(error)
                    else:
                        _LOGGER.exception("Unexpected web research search failure.")
                        safe_message = (
                            "The search could not be completed. Try again later."
                        )
                    query_attempts[query_key] = (None, safe_message)
            results, query_error = query_attempts[query_key]
            if query_error is not None:
                candidates.append(
                    CandidateResearch(
                        co_id=generated_query.co_id,
                        co_description=outcomes[generated_query.co_id],
                        search_query=safe_query,
                        status="search_failed",
                        result_count=0,
                        sources=(),
                        error=query_error,
                    )
                )
                continue

            if self._is_cancelled(cancellation_event):
                candidates.append(
                    self._candidate_cancelled(
                        generated_query, outcomes[generated_query.co_id]
                    )
                )
                continue

            if results is None:
                results = ()
            ranked = self._rank_results(
                safe_generated_query,
                results,
                academic_question=academic_question,
                co_description=outcomes[generated_query.co_id],
                technical_concepts=relevant_technical_concepts,
                query_analysis=analysis,
            )
            unique_ranked = self._deduplicate_candidate_results(ranked)
            kept = tuple(unique_ranked[: self.max_sources_per_co])
            collected.extend(kept)
            candidates.append(
                CandidateResearch(
                    co_id=generated_query.co_id,
                    co_description=outcomes[generated_query.co_id],
                    search_query=safe_query,
                    status="success" if kept else "empty",
                    result_count=len(results),
                    sources=(),
                )
            )

        if self._is_cancelled(cancellation_event):
            return self._result(
                "cancelled",
                analysis,
                tuple(candidates),
                (),
                queries_attempted,
                "Web research was cancelled.",
            )

        extracted = self._extract_unique_sources(collected, cancellation_event)
        source_map = {
            self._url_key(source.url): source for source in extracted
        }
        sources = tuple(
            self._to_source(item, source_map.get(self._url_key(item.search_result.url)))
            for item in collected
        )
        by_candidate: dict[str, list[ResearchSource]] = {}
        for source in sources:
            by_candidate.setdefault(source.candidate_co_id, []).append(source)
        candidates = [
            CandidateResearch(
                co_id=candidate.co_id,
                co_description=candidate.co_description,
                search_query=candidate.search_query,
                status=candidate.status,
                result_count=candidate.result_count,
                sources=tuple(by_candidate.get(candidate.co_id, [])),
                error=candidate.error,
            )
            for candidate in candidates
        ]

        failed_count = sum(
            candidate.status == "search_failed" for candidate in candidates
        )
        successful_count = sum(
            candidate.status == "success" for candidate in candidates
        )
        if self._is_cancelled(cancellation_event):
            status: ResearchStatus = "cancelled"
            message = "Web research was cancelled."
        elif failed_count == len(candidates):
            status = "search_unavailable"
            message = "Web search is currently unavailable."
        elif failed_count or (successful_count and successful_count < len(candidates)):
            status = "partial"
            message = "Some CO-specific searches failed or returned no usable sources."
        elif queries_attempted == 0:
            status = "no_public_queries"
            message = (
                "No public-safe technical concepts were available for web search."
            )
        elif successful_count:
            status = "success"
            message = None
        else:
            status = "success"
            message = "Search completed, but no sufficiently relevant sources were found."

        return self._result(
            status,
            analysis,
            tuple(candidates),
            sources,
            queries_attempted,
            message,
        )

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._search_cache.clear()
        self.extraction_service.clear_cache()

    def _search(self, query: str) -> tuple[SearchResult, ...]:
        cache_key = preprocess_text(query)
        now = self._clock()
        with self._cache_lock:
            entry = self._search_cache.get(cache_key)
            if entry is not None and entry.expires_at > now:
                self._search_cache.move_to_end(cache_key)
                return entry.results
            if entry is not None:
                del self._search_cache[cache_key]

        results = self.search_service.search(
            query,
            max_results=self.results_per_query,
        )
        if not isinstance(results, list):
            raise SearchResponseError(
                "The search provider returned an unexpected response."
            )
        cached_results = tuple(results)
        with self._cache_lock:
            self._search_cache[cache_key] = _SearchCacheEntry(
                cached_results, self._clock() + self.cache_seconds
            )
            self._search_cache.move_to_end(cache_key)
            while len(self._search_cache) > self.max_cache_entries:
                self._search_cache.popitem(last=False)
        return cached_results

    def _rank_results(
        self,
        query: GeneratedSearchQuery,
        results: Sequence[SearchResult],
        *,
        academic_question: str,
        co_description: str,
        technical_concepts: Sequence[str],
        query_analysis: SyllabusAwareQueryResult,
    ) -> list[_ScoredResult]:
        relevant_topics = [
            match.topic
            for match in query_analysis.relevant_syllabus_topics
        ]
        context = " ".join(
            [
                academic_question,
                co_description,
                *technical_concepts,
                *relevant_topics,
            ]
        )
        valid_results: list[SearchResult] = []
        valid_urls: list[str] = []
        for result in results:
            if not isinstance(result, SearchResult):
                continue
            try:
                url = normalize_http_url(result.url)
            except InvalidResultURLError:
                continue
            valid_results.append(result)
            valid_urls.append(url)

        result_texts = [
            f"{result.title} {result.snippet}" for result in valid_results
        ]
        tfidf_scores = tfidf_similarity_scores(context, result_texts)
        scored: list[_ScoredResult] = []
        for result, url, searchable_text, tfidf in zip(
            valid_results,
            valid_urls,
            result_texts,
            tfidf_scores,
            strict=True,
        ):
            context_terms = self._terms(context)
            result_terms = self._terms(searchable_text)
            overlap = (
                len(context_terms.intersection(result_terms)) / len(context_terms)
                if context_terms
                else 0.0
            )
            relevance = 0.8 * tfidf + 0.2 * overlap
            if relevance < self.relevance_threshold:
                continue
            normalized = SearchResult(
                title=result.title,
                url=url,
                snippet=result.snippet,
                source=result.source,
                metadata=result.metadata,
            )
            scored.append(
                _ScoredResult(
                    candidate_co_id=query.co_id,
                    query=query,
                    search_result=normalized,
                    relevance_score=relevance,
                    source_quality=self._source_quality(url),
                )
            )
        scored.sort(
            key=lambda item: (
                -(
                    item.relevance_score
                    + 0.015 * self._quality_priority(item.source_quality)
                ),
                -self._quality_priority(item.source_quality),
                item.search_result.url,
            )
        )
        return scored

    @staticmethod
    def _privacy_safe_query(
        technical_concepts: Sequence[str],
        cognitive_action: str,
        co_description: str,
        relevant_syllabus_topics: Sequence[str] = (),
    ) -> str:
        allowed_terms = ResearchEngine._terms(co_description)
        for topic in relevant_syllabus_topics:
            allowed_terms.update(ResearchEngine._terms(topic))
        terms = [
            concept.strip()
            for concept in technical_concepts
            if isinstance(concept, str)
            and concept.strip()
            and 1 <= len(ResearchEngine._terms(concept)) <= 3
            and len(concept) <= 48
            and ResearchEngine._terms(concept).issubset(allowed_terms)
        ]
        safe_parts = ResearchEngine._deduplicate_text(terms)
        if not safe_parts:
            return ""
        safe_parts.extend(
            [cognitive_action.strip(), "academic technical concepts"]
        )
        return " ".join(ResearchEngine._deduplicate_text(safe_parts))[:500]

    @staticmethod
    def _deduplicate_text(values: Sequence[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = preprocess_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(value.strip())
        return unique

    @staticmethod
    def _deduplicate_candidate_results(
        results: Sequence[_ScoredResult],
    ) -> list[_ScoredResult]:
        unique: list[_ScoredResult] = []
        seen: set[str] = set()
        for result in results:
            key = ResearchEngine._url_key(result.search_result.url)
            if key not in seen:
                seen.add(key)
                unique.append(result)
        return unique

    def _extract_unique_sources(
        self,
        collected: Sequence[_ScoredResult],
        cancellation_event: Event | None,
    ) -> tuple[WebContentResult, ...]:
        unique: list[_ScoredResult] = []
        seen: set[str] = set()
        for item in sorted(
            collected,
            key=lambda result: (
                -(
                    result.relevance_score
                    + 0.015 * self._quality_priority(result.source_quality)
                ),
                -self._quality_priority(result.source_quality),
            ),
        ):
            key = self._url_key(item.search_result.url)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        pages_to_extract = unique[: self.max_pages_to_extract]
        if not pages_to_extract:
            return ()
        if self._is_cancelled(cancellation_event):
            return tuple(
                WebContentResult(
                    title=item.search_result.title,
                    url=item.search_result.url,
                    snippet=item.search_result.snippet,
                    content=item.search_result.snippet,
                    source=item.search_result.source,
                    metadata=dict(item.search_result.metadata),
                    publication_date=None,
                    retrieved_at=self._timestamp(),
                    extraction_status="cancelled",
                )
                for item in pages_to_extract
            )

        try:
            return tuple(
                self.extraction_service.extract_results(
                    [item.search_result for item in pages_to_extract],
                    cancellation_event=cancellation_event,
                )
            )
        except Exception:
            _LOGGER.exception("Webpage evidence extraction failed.")
            return tuple(
                WebContentResult(
                    title=item.search_result.title,
                    url=item.search_result.url,
                    snippet=item.search_result.snippet,
                    content=item.search_result.snippet,
                    source=item.search_result.source,
                    metadata=dict(item.search_result.metadata),
                    publication_date=None,
                    retrieved_at=self._timestamp(),
                    extraction_status="failed",
                    error_details="Webpage extraction could not be completed.",
                )
                for item in pages_to_extract
            )

    def _to_source(
        self,
        item: _ScoredResult,
        extracted: WebContentResult | None,
    ) -> ResearchSource:
        if extracted is None:
            extraction_status: ExtractionStatus = "not_attempted"
            content = None
            publication_date = item.search_result.metadata.get("date")
            if not isinstance(publication_date, str):
                publication_date = None
            retrieved_at = self._timestamp()
            error_details = None
        else:
            extraction_status = extracted.extraction_status
            content = (
                extracted.content
                if extracted.extraction_status in {"success", "cached"}
                else None
            )
            publication_date = extracted.publication_date
            if publication_date is None:
                candidate_date = item.search_result.metadata.get("date")
                if isinstance(candidate_date, str):
                    publication_date = candidate_date
            retrieved_at = extracted.retrieved_at or self._timestamp()
            error_details = extracted.error_details
        return ResearchSource(
            candidate_co_id=item.candidate_co_id,
            search_query=item.query.query,
            title=item.search_result.title,
            url=item.search_result.url,
            search_snippet=item.search_result.snippet,
            extracted_content=content,
            publication_date=publication_date,
            retrieved_at=retrieved_at,
            search_provider="DDGS",
            extraction_status=extraction_status,
            relevance_score=item.relevance_score,
            source_quality=item.source_quality,
            error_details=error_details,
        )

    @staticmethod
    def _source_quality(url: str) -> SourceQuality:
        host = (urlsplit(url).hostname or "").casefold().rstrip(".")
        labels = host.split(".")
        if host.endswith(".edu") or (
            len(labels) >= 3 and labels[-2] == "ac" and len(labels[-1]) == 2
        ):
            return "university"
        if any(
            host == domain or host.endswith(f".{domain}")
            for domain in _DOCUMENTATION_DOMAINS
        ):
            return "official_documentation"
        if any(
            host == domain or host.endswith(f".{domain}")
            for domain in _EDUCATIONAL_DOMAINS
        ):
            return "academic" if host in {"arxiv.org", "doi.org"} else "educational"
        return "general"

    @staticmethod
    def _quality_priority(quality: SourceQuality) -> int:
        return {
            "university": 4,
            "official_documentation": 3,
            "academic": 3,
            "educational": 2,
            "general": 1,
        }[quality]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(_TOKEN_PATTERN.findall(preprocess_text(text)))

    @staticmethod
    def _url_key(url: str) -> str:
        try:
            return normalize_http_url(url)
        except InvalidResultURLError:
            return url.strip().casefold()

    @staticmethod
    def _candidate_cancelled(
        query: GeneratedSearchQuery, description: str
    ) -> CandidateResearch:
        return CandidateResearch(
            co_id=query.co_id,
            co_description=description,
            search_query=query.query,
            status="cancelled",
            result_count=0,
            sources=(),
        )

    @staticmethod
    def _is_cancelled(event: Event | None) -> bool:
        return event is not None and event.is_set()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @classmethod
    def _result(
        cls,
        status: ResearchStatus,
        analysis: SyllabusAwareQueryResult,
        candidates: tuple[CandidateResearch, ...],
        sources: tuple[ResearchSource, ...],
        query_count: int,
        message: str | None,
    ) -> ResearchEngineResult:
        return ResearchEngineResult(
            status=status,
            query_analysis=analysis,
            candidates=candidates,
            sources=sources,
            search_query_count=query_count,
            retrieved_at=cls._timestamp(),
            message=message,
        )

    @staticmethod
    def _validate_limit(value: int, maximum: int, label: str) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise ValueError(f"{label} must be between 1 and {maximum}.")
        return value

    @staticmethod
    def _validate_positive_int(value: int, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer.")
        return value

    @staticmethod
    def _validate_threshold(value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("Relevance threshold must be between 0 and 1.")
        return float(value)
