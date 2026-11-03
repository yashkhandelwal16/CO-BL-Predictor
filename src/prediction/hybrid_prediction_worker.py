import logging
import re
from threading import Event
from typing import Any

from PySide6.QtCore import QThread, Signal

from src.ai.llm_client import (
    GenerationCancelledError,
    InferenceError,
    InferenceTimeoutError,
    LLMClient,
    ModelConfigurationError,
)
from src.database.db_manager import DatabaseError, DatabaseManager
from src.nlp.text_preprocessor import ApprovedExampleRetriever
from src.prediction.bloom_rules import (
    load_bloom_config,
)
from src.prediction.hybrid_prediction_engine import (
    HybridPredictionEngine,
    HybridPredictionResult,
)
from src.prediction.predictor import (
    PredictionInputError,
    PredictionModelError,
    QuestionPredictor,
)
from src.research.academic_evidence_analyzer import (
    AcademicEvidenceAnalysisError,
    AcademicEvidenceAnalyzer,
)
from src.research.research_engine import ResearchEngine, ResearchEngineResult
from src.research.syllabus_query_generator import SyllabusAwareQueryGenerator
from src.research.web_search import WebSearchError


_LOGGER = logging.getLogger(__name__)
_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_STOP_WORDS = frozenset(
    {
        "a", "about", "after", "all", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "between", "both", "but",
        "by", "can", "could", "describe", "determine", "discuss", "do", "does",
        "during", "each", "evaluate", "explain", "for", "from", "give", "has",
        "have", "how", "identify", "if", "in", "into", "is", "it", "its",
        "may", "might", "must", "of", "on", "or", "should", "show", "solve",
        "state", "that", "the", "their", "then", "there", "these", "this",
        "those", "through", "to", "under", "use", "using", "was", "were",
        "what", "when", "where", "which", "who", "why", "will", "with",
        "would", "write", "analyze", "apply", "calculate", "classify",
        "compare", "compute", "create", "design", "derive", "develop",
        "formulate", "construct", "contrast", "demonstrate", "differentiate",
        "examine", "justify", "list", "recall", "summarize", "verify",
    }
)
_ACTION_VERBS = (
    "design", "create", "construct", "formulate", "evaluate", "assess",
    "justify", "critique", "analyze", "analyse", "compare", "differentiate",
    "examine", "calculate", "compute", "solve", "apply", "implement",
    "demonstrate", "classify", "interpret", "explain", "describe", "summarize",
    "summarise", "identify", "define", "list", "state", "recall",
)


class HybridPredictionWorker(QThread):
    """Run local prediction, optional web research, and evidence comparison."""

    completed = Signal(str, str)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(str)

    def __init__(
        self,
        database: DatabaseManager,
        course_id: str,
        question_text: str,
        bloom_config: dict[str, Any] | None = None,
        *,
        web_research_enabled: bool = False,
    ) -> None:
        super().__init__()
        self._database = database
        self._course_id = course_id
        self._question_text = question_text
        self._bloom_config = bloom_config or load_bloom_config()
        self._web_research_enabled = web_research_enabled
        self._cancellation_event = Event()

    def cancel(self) -> None:
        self._cancellation_event.set()

    def run(self) -> None:
        try:
            result, status_message = self._run_pipeline()
        except PredictionInputError as error:
            self.failed.emit(str(error))
        except GenerationCancelledError:
            self.cancelled.emit()
        except InferenceTimeoutError:
            self.failed.emit(
                "Classification took too long. Try a shorter question and try again."
            )
        except InferenceError:
            _LOGGER.exception("Local model inference failed.")
            self.failed.emit(
                "Local classification could not be completed. Check the model "
                "setup and try again."
            )
        except (PredictionModelError, ModelConfigurationError, DatabaseError):
            _LOGGER.exception("Hybrid prediction could not be completed.")
            self.failed.emit(
                "Classification could not be completed. Check the local "
                "application files and try again."
            )
        except (ValueError, KeyError, TypeError):
            _LOGGER.exception("Hybrid prediction received invalid course evidence.")
            self.failed.emit(
                "Classification could not be completed with the selected course data."
            )
        else:
            self.completed.emit(result.format_output(), status_message)

    def _run_pipeline(self) -> tuple[HybridPredictionResult, str]:
        self._check_cancelled()
        llm_client = LLMClient()
        predictor = QuestionPredictor(
            self._database,
            llm_client,
            self._bloom_config,
        )
        self.progress.emit("Reviewing the question and course context...")
        local_prediction = predictor.predict(
            self._course_id,
            self._question_text,
            cancellation_event=self._cancellation_event,
        )
        self._check_cancelled()

        course = self._database.get_course_by_id(self._course_id)
        if course is None:
            raise PredictionInputError(
                "The selected course is no longer available."
            )
        outcomes = course["course_outcomes"]
        syllabus_versions = self._database.list_syllabus_versions(
            course_id=course["id"]
        )
        reviewed_syllabus = next(
            (
                syllabus
                for syllabus in syllabus_versions
                if syllabus["review_status"] == "reviewed"
            ),
            None,
        )
        units = (
            reviewed_syllabus["content"].get("units", [])
            if reviewed_syllabus is not None
            else []
        )
        concepts = self._technical_concepts(
            self._question_text, outcomes, units
        )
        action = self._detect_cognitive_action(self._question_text)
        query_generator = SyllabusAwareQueryGenerator()
        query_analysis = query_generator.generate(
            academic_question=self._question_text,
            course_name=course["course_name"],
            course_code=course["course_code"],
            course_outcomes=outcomes,
            unit_topics=units,
            relevant_technical_concepts=concepts,
            detected_cognitive_action=action,
        )
        approved_examples = ApprovedExampleRetriever(
            self._database
        ).retrieve(
            course["course_code"],
            self._question_text,
            bl_scheme=None,
        )

        research_result: ResearchEngineResult | None = None
        academic_analysis = None
        research_has_sources = False
        if self._web_research_enabled:
            self._check_cancelled()
            self.progress.emit("Searching optional public technical sources...")
            try:
                research_result = ResearchEngine(
                    max_queries=3,
                    results_per_query=5,
                    max_sources_per_co=3,
                    max_pages_to_extract=2,
                ).research(
                    academic_question=self._question_text,
                    course_name=course["course_name"],
                    course_code=course["course_code"],
                    course_outcomes=outcomes,
                    unit_topics=units,
                    relevant_technical_concepts=concepts,
                    detected_cognitive_action=action,
                    cancellation_event=self._cancellation_event,
                )
                research_has_sources = bool(research_result.sources)
            except WebSearchError:
                _LOGGER.exception("Optional web research failed; continuing locally.")
                research_result = None

            self._check_cancelled()
            self.progress.emit("Comparing available academic evidence...")
            try:
                academic_analysis = AcademicEvidenceAnalyzer(
                    llm_client=llm_client,
                    max_sources=3,
                    max_tokens=700,
                ).analyze(
                    academic_question=self._question_text,
                    course_name=course["course_name"],
                    course_code=course["course_code"],
                    official_course_outcomes=outcomes,
                    relevant_syllabus_units=units,
                    nlp_candidate_co_rankings=query_analysis,
                    ddgs_search_results=(
                        research_result.sources if research_result else ()
                    ),
                    faculty_approved_examples=approved_examples,
                    detected_cognitive_action=action,
                    cancellation_event=self._cancellation_event,
                )
            except GenerationCancelledError:
                raise
            except AcademicEvidenceAnalysisError:
                _LOGGER.exception(
                    "Academic evidence analysis failed; using local prediction."
                )
                academic_analysis = None
                research_result = None
                research_has_sources = False

        self._check_cancelled()
        self.progress.emit("Finalizing the suggestion...")
        hybrid = HybridPredictionEngine().predict(
            question_text=self._question_text,
            official_course_outcomes=outcomes,
            syllabus_units=units,
            local_prediction=local_prediction,
            nlp_candidate_co_rankings=query_analysis,
            web_research=research_result,
            academic_analysis=academic_analysis,
            faculty_approved_examples=approved_examples,
        )
        self._database.save_hybrid_prediction(
            local_prediction.prediction_id,
            hybrid.co_label,
            hybrid.bl_label,
            status=(
                "insufficient_context"
                if hybrid.status != "predicted"
                else "predicted"
            ),
            pipeline_mode=hybrid.decision_record.pipeline_mode,
            bl_scheme=hybrid.bl_scheme,
            decision_record={
                "selected_co": hybrid.decision_record.selected_co,
                "selected_bl": hybrid.decision_record.selected_bl,
                "supporting_evidence": list(
                    hybrid.decision_record.supporting_evidence
                ),
                "conflicting_evidence": list(
                    hybrid.decision_record.conflicting_evidence
                ),
                "sources_used": [
                    {
                        "source_id": source.source_id,
                        "title": source.title,
                        "url": source.url,
                        "candidate_co_id": source.candidate_co_id,
                        "search_query": source.search_query,
                        "search_snippet": source.search_snippet,
                        "extracted_content": source.extracted_content,
                        "publication_date": source.publication_date,
                        "retrieved_at": source.retrieved_at,
                        "search_provider": source.search_provider,
                        "extraction_status": source.extraction_status,
                        "source_quality": source.source_quality,
                    }
                    for source in hybrid.decision_record.sources_used
                ],
                "co_abstention_reason": (
                    hybrid.decision_record.co_abstention_reason
                ),
                "bl_abstention_reason": (
                    hybrid.decision_record.bl_abstention_reason
                ),
                "pipeline_mode": hybrid.decision_record.pipeline_mode,
            },
        )

        if not self._web_research_enabled:
            status_message = (
                "Local-only suggestion ready; optional public research is off."
            )
        elif research_has_sources and academic_analysis is not None:
            status_message = "Hybrid suggestion ready for faculty review."
        else:
            status_message = (
                "Local-only suggestion ready; public research was unavailable "
                "or returned no usable sources."
            )
        return hybrid, status_message

    def _check_cancelled(self) -> None:
        if self._cancellation_event.is_set():
            raise GenerationCancelledError(
                "Hybrid prediction was cancelled."
            )

    @staticmethod
    def _technical_concepts(
        question: str,
        outcomes: dict[str, str],
        units: list[dict[str, Any]],
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
                        *outcomes.values(),
                        *(
                            str(topic)
                            for unit in units
                            if isinstance(unit, dict)
                            for topic in unit.get("topics", [])
                        ),
                    ]
                )
            )
            if len(token) > 2 and token.casefold() not in _STOP_WORDS
        }
        return tuple(sorted(question_terms.intersection(course_terms)))[:12]

    @staticmethod
    def _detect_cognitive_action(question: str) -> str:
        terms = {
            token.casefold() for token in _TOKEN_PATTERN.findall(question)
        }
        return next(
            (action for action in _ACTION_VERBS if action in terms),
            "not specified",
        )
