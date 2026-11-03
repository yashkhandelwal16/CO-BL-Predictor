import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Event

from src.prediction.bloom_taxonomy import (
    classify_bloom_level,
    detect_cognitive_action,
)
from src.research.research_engine import ResearchEngine, ResearchEngineResult
from src.research.syllabus_query_generator import SyllabusAwareQueryGenerator


_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
        "describe", "determine", "do", "does", "during", "evaluate", "explain",
        "for", "from", "give", "has", "have", "how", "identify", "if", "in",
        "into", "is", "it", "its", "may", "might", "must", "of", "on", "or",
        "should", "show", "solve", "state", "that", "the", "their", "these",
        "this", "those", "through", "to", "under", "use", "using", "was",
        "were", "what", "when", "where", "which", "who", "why", "will",
        "with", "would", "write",
    }
)
MIN_CO_EVIDENCE_SCORE = 0.05
MIN_CO_EVIDENCE_MARGIN = 0.01


@dataclass(frozen=True)
class WebSearchPredictionResult:
    co_label: str
    bl_label: str
    status_message: str

    def format_output(self) -> str:
        return f"CO level - {self.co_label}\nBL level - {self.bl_label}"


class WebSearchPredictionService:
    """Make independent, non-persisted CO/BL suggestions from public evidence."""

    def __init__(self, research_engine: ResearchEngine | None = None) -> None:
        self.research_engine = research_engine or ResearchEngine(
            query_generator=SyllabusAwareQueryGenerator(
                min_co_similarity=0.001
            ),
            max_queries=5,
            results_per_query=5,
            max_sources_per_co=3,
            max_pages_to_extract=2,
        )

    def course_context_suggestion(
        self,
        *,
        question: str,
        course_name: str,
        course_code: str,
        course_outcomes: Mapping[str, str],
        syllabus_units: (
            Mapping[str, Sequence[str]]
            | Sequence[Mapping[str, object]]
        ),
    ) -> str:
        concepts = self._technical_concepts(
            question, course_outcomes, syllabus_units
        )
        rankings = SyllabusAwareQueryGenerator().generate(
            academic_question=question,
            course_name=course_name,
            course_code=course_code,
            course_outcomes=course_outcomes,
            unit_topics=syllabus_units,
            relevant_technical_concepts=concepts,
            detected_cognitive_action=detect_cognitive_action(question),
        ).co_rankings
        return self._best_context_match(
            {ranking.co_id: ranking.similarity for ranking in rankings},
            set(course_outcomes),
        ) or "N/A"

    def predict(
        self,
        *,
        question: str,
        course_name: str,
        course_code: str,
        course_outcomes: Mapping[str, str],
        syllabus_units: (
            Mapping[str, Sequence[str]]
            | Sequence[Mapping[str, object]]
        ),
        cancellation_event: Event | None = None,
    ) -> WebSearchPredictionResult:
        if not question.strip():
            return WebSearchPredictionResult(
                "N/A", "N/A", "Enter an examination question to search."
            )

        concepts = self._technical_concepts(
            question,
            course_outcomes,
            syllabus_units,
        )
        research = self.research_engine.research(
            academic_question=question,
            course_name=course_name,
            course_code=course_code,
            course_outcomes=course_outcomes,
            unit_topics=syllabus_units,
            relevant_technical_concepts=concepts,
            detected_cognitive_action=detect_cognitive_action(question),
            cancellation_event=cancellation_event,
        )
        bl_label = classify_bloom_level(question)
        co_label, used_fallback = self._select_course_outcome(
            research,
            question=question,
            course_name=course_name,
            course_code=course_code,
            course_outcomes=course_outcomes,
            syllabus_units=syllabus_units,
            concepts=concepts,
        )
        return WebSearchPredictionResult(
            co_label=co_label,
            bl_label=bl_label,
            status_message=self._status_message(
                research, co_label, used_fallback
            ),
        )

    @classmethod
    def _select_course_outcome(
        cls,
        research: ResearchEngineResult,
        *,
        question: str,
        course_name: str,
        course_code: str,
        course_outcomes: Mapping[str, str],
        syllabus_units: (
            Mapping[str, Sequence[str]]
            | Sequence[Mapping[str, object]]
        ),
        concepts: Sequence[str],
    ) -> tuple[str, bool]:
        if research.status == "cancelled":
            return "N/A", False

        scores: dict[str, float] = {}
        for source in research.sources:
            if source.candidate_co_id in course_outcomes:
                scores[source.candidate_co_id] = max(
                    scores.get(source.candidate_co_id, 0.0),
                    source.relevance_score,
                )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        query_analysis = getattr(research, "query_analysis", None)
        rankings = getattr(query_analysis, "co_rankings", ())
        if not rankings:
            rankings = SyllabusAwareQueryGenerator().generate(
                academic_question=question,
                course_name=course_name,
                course_code=course_code,
                course_outcomes=course_outcomes,
                unit_topics=syllabus_units,
                relevant_technical_concepts=concepts,
                detected_cognitive_action=detect_cognitive_action(question),
            ).co_rankings
        context_scores = {
            ranking.co_id: ranking.similarity
            for ranking in rankings
            if ranking.co_id in course_outcomes
        }

        if ranked:
            top_co, top_score = ranked[0]
            margin = (
                top_score - ranked[1][1]
                if len(ranked) > 1
                else float("inf")
            )
            if (
                top_score >= MIN_CO_EVIDENCE_SCORE
                and margin >= MIN_CO_EVIDENCE_MARGIN
            ):
                return top_co, False

            if margin < MIN_CO_EVIDENCE_MARGIN:
                tied_candidates = {
                    co_id
                    for co_id, score in ranked
                    if top_score - score < MIN_CO_EVIDENCE_MARGIN
                }
                context_choice = cls._best_context_match(
                    context_scores, tied_candidates
                )
                if context_choice is not None:
                    return context_choice, True
            return top_co, True

        context_choice = cls._best_context_match(
            context_scores, set(course_outcomes)
        )
        return (
            (context_choice, True)
            if context_choice is not None
            else ("N/A", False)
        )

    @staticmethod
    def _best_context_match(
        scores: Mapping[str, float], candidates: set[str]
    ) -> str | None:
        ranked = sorted(
            (
                (co_id, score)
                for co_id, score in scores.items()
                if co_id in candidates
            ),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[0][0] if ranked and ranked[0][1] > 0 else None

    @staticmethod
    def _status_message(
        research: ResearchEngineResult,
        co_label: str,
        used_fallback: bool,
    ) -> str:
        if research.status == "search_unavailable":
            if co_label != "N/A":
                return (
                    "Web search is unavailable; showing the best syllabus-based "
                    "CO match for faculty review."
                )
            return research.message or (
                "Web search is unavailable. Check your internet connection."
            )
        if research.status == "cancelled":
            return "Web search was cancelled."
        if research.status == "partial":
            if used_fallback and co_label != "N/A":
                return (
                    "Some searches were unavailable; showing the best "
                    "syllabus-based CO match for faculty review."
                )
            return (
                "Some searches were unavailable; the Web Search result may "
                "have incomplete evidence."
            )
        if used_fallback and co_label != "N/A":
            return (
                "No clear web-supported CO match; showing the best "
                "syllabus-based CO match for faculty review."
            )
        if co_label == "N/A":
            return research.message or (
                "Web search found no clear evidence for one course outcome."
            )
        return "Web Search suggestion ready for faculty review."

    @staticmethod
    def _technical_concepts(
        question: str,
        outcomes: Mapping[str, str],
        units: (
            Mapping[str, Sequence[str]]
            | Sequence[Mapping[str, object]]
        ),
    ) -> tuple[str, ...]:
        question_terms = {
            token.casefold()
            for token in _TOKEN_PATTERN.findall(question)
            if len(token) > 2 and token.casefold() not in _STOP_WORDS
        }
        course_terms = {
            token.casefold()
            for token in _TOKEN_PATTERN.findall(
                " ".join(
                    [
                        *(str(description) for description in outcomes.values()),
                        *WebSearchPredictionService._unit_text(units),
                    ]
                )
            )
            if len(token) > 2 and token.casefold() not in _STOP_WORDS
        }
        return tuple(sorted(question_terms.intersection(course_terms)))[:12]

    @staticmethod
    def _unit_text(
        units: (
            Mapping[str, Sequence[str]]
            | Sequence[Mapping[str, object]]
        ),
    ) -> list[str]:
        if isinstance(units, Mapping):
            return [
                f"{unit_name} {' '.join(str(topic) for topic in topics)}"
                for unit_name, topics in units.items()
            ]
        text: list[str] = []
        for unit in units:
            if not isinstance(unit, Mapping):
                continue
            text.append(str(unit.get("unit_title", "")))
            topics = unit.get("topics", [])
            if isinstance(topics, Sequence) and not isinstance(
                topics, (str, bytes)
            ):
                text.extend(str(topic) for topic in topics)
        return text
