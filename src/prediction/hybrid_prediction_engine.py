import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from src.nlp.text_preprocessor import SimilarApprovedExample
from src.prediction.predictor import PredictionResult
from src.research.academic_evidence_analyzer import (
    AcademicEvidenceReport,
    EvidenceSourceReference,
)
from src.research.research_engine import ResearchEngineResult, ResearchSource
from src.research.syllabus_query_generator import (
    CourseOutcomeRanking,
    SyllabusAwareQueryResult,
)


_BL_LEVELS = frozenset({"BL1", "BL2", "BL3", "BL4", "BL5", "BL6"})
_LEGACY_BL_LEVELS = frozenset({"BL1", "BL2", "BL3", "BL4", "BL5"})
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

PipelineMode = Literal["hybrid", "local-only"]
PredictionStatus = Literal["predicted", "insufficient_context", "invalid_question"]
BloomScheme = Literal["custom_bl0_bl5", "revised_bl1_bl6"]


@dataclass(frozen=True)
class DecisionSource:
    source_id: str
    title: str
    url: str
    candidate_co_id: str
    search_query: str | None = None
    search_snippet: str | None = None
    extracted_content: str | None = None
    publication_date: str | None = None
    retrieved_at: str | None = None
    search_provider: str | None = None
    extraction_status: str | None = None
    source_quality: str | None = None


@dataclass(frozen=True)
class HybridDecisionRecord:
    selected_co: str
    selected_bl: str
    supporting_evidence: tuple[str, ...]
    conflicting_evidence: tuple[str, ...]
    sources_used: tuple[DecisionSource, ...]
    co_abstention_reason: str | None
    bl_abstention_reason: str | None
    pipeline_mode: PipelineMode


@dataclass(frozen=True)
class HybridPredictionResult:
    co_label: str
    bl_label: str
    status: PredictionStatus
    bl_scheme: BloomScheme
    decision_record: HybridDecisionRecord

    def format_output(self) -> str:
        return f"CO level - {self.co_label}\nBL level - {self.bl_label}"


@dataclass(frozen=True)
class _PreparedEvidence:
    outcomes: dict[str, str]
    units: tuple[tuple[str, tuple[str, ...]], ...]
    rankings: tuple[CourseOutcomeRanking, ...]
    examples: tuple[SimilarApprovedExample, ...]
    research_sources: tuple[ResearchSource, ...]
    source_catalog: tuple[EvidenceSourceReference, ...]
    academic_analysis: AcademicEvidenceReport | None


class HybridPredictionEngine:
    """Compare prediction evidence without replacing the existing predictor."""

    def __init__(
        self,
        *,
        min_co_similarity: float = 0.08,
        co_decision_margin: float = 0.04,
        min_example_similarity: float = 0.15,
    ) -> None:
        for value, name in (
            (min_co_similarity, "Minimum CO similarity"),
            (co_decision_margin, "CO decision margin"),
            (min_example_similarity, "Minimum example similarity"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be between 0 and 1.")
        self.min_co_similarity = float(min_co_similarity)
        self.co_decision_margin = float(co_decision_margin)
        self.min_example_similarity = float(min_example_similarity)

    def predict(
        self,
        question_text: str,
        official_course_outcomes: Mapping[str, str],
        syllabus_units: (
            Mapping[str, Sequence[str]]
            | Sequence[Mapping[str, object]]
        ),
        *,
        local_prediction: PredictionResult | None = None,
        nlp_candidate_co_rankings: (
            SyllabusAwareQueryResult | Sequence[CourseOutcomeRanking] | None
        ) = None,
        web_research: ResearchEngineResult | Sequence[ResearchSource] | None = None,
        academic_analysis: AcademicEvidenceReport | None = None,
        faculty_approved_examples: (
            Sequence[SimilarApprovedExample] | None
        ) = None,
    ) -> HybridPredictionResult:
        question = self._required_text(question_text, "Question")
        prepared = self._prepare_evidence(
            official_course_outcomes,
            syllabus_units,
            nlp_candidate_co_rankings,
            web_research,
            faculty_approved_examples,
            academic_analysis,
        )
        self._validate_local_prediction(local_prediction, prepared.outcomes)
        mode: PipelineMode = (
            "hybrid"
            if (
                prepared.research_sources
                or (
                    academic_analysis is not None
                    and academic_analysis.source_catalog
                )
            )
            else "local-only"
        )

        if not self._is_course_relevant(question, prepared):
            record = HybridDecisionRecord(
                selected_co="N/A",
                selected_bl="N/A",
                supporting_evidence=(),
                conflicting_evidence=(),
                sources_used=(),
                co_abstention_reason=(
                    "The question is empty, non-academic, or has no meaningful "
                    "overlap with the supplied official course context."
                ),
                bl_abstention_reason=(
                    "Cognitive demand cannot be classified for a question "
                    "outside the supplied course context."
                ),
                pipeline_mode=mode,
            )
            return HybridPredictionResult(
                co_label="N/A",
                bl_label="N/A",
                status="invalid_question",
                bl_scheme=(
                    "revised_bl1_bl6"
                    if academic_analysis is not None
                    or any(
                        example.bl_scheme == "revised_bl1_bl6"
                        and example.similarity >= self.min_example_similarity
                        for example in prepared.examples
                    )
                    else "custom_bl0_bl5"
                ),
                decision_record=record,
            )

        supports: list[str] = []
        conflicts: list[str] = []
        source_ids: set[str] = set()
        co_label, co_abstention = self._decide_co(
            prepared,
            local_prediction,
            academic_analysis,
            supports,
            conflicts,
            source_ids,
        )
        bl_label, bl_abstention = self._decide_bl(
            prepared,
            local_prediction,
            academic_analysis,
            supports,
            conflicts,
            source_ids,
        )
        sources = self._sources_used(prepared, source_ids)
        status: PredictionStatus = (
            "insufficient_context"
            if co_label == "N/A" or bl_label == "N/A"
            else "predicted"
        )
        record = HybridDecisionRecord(
            selected_co=co_label,
            selected_bl=bl_label,
            supporting_evidence=tuple(dict.fromkeys(supports)),
            conflicting_evidence=tuple(dict.fromkeys(conflicts)),
            sources_used=sources,
            co_abstention_reason=co_abstention,
            bl_abstention_reason=bl_abstention,
            pipeline_mode=mode,
        )
        return HybridPredictionResult(
            co_label=co_label,
            bl_label=bl_label,
            status=status,
            bl_scheme=(
                "revised_bl1_bl6"
                if academic_analysis is not None
                or any(
                    example.bl_scheme == "revised_bl1_bl6"
                    and example.similarity >= self.min_example_similarity
                    for example in prepared.examples
                )
                else "custom_bl0_bl5"
            ),
            decision_record=record,
        )

    def _decide_co(
        self,
        evidence: _PreparedEvidence,
        local_prediction: PredictionResult | None,
        analysis: AcademicEvidenceReport | None,
        supports: list[str],
        conflicts: list[str],
        source_ids: set[str],
    ) -> tuple[str, str | None]:
        ranked = sorted(
            evidence.rankings,
            key=lambda item: (-item.similarity, item.co_id),
        )
        eligible = [
            ranking
            for ranking in ranked
            if ranking.similarity >= self.min_co_similarity
        ]
        top = eligible[0] if eligible else None
        next_score = eligible[1].similarity if len(eligible) > 1 else 0.0
        nlp_contested = (
            top is not None
            and len(eligible) > 1
            and top.similarity - next_score < self.co_decision_margin
        )
        nlp_decisive = top is not None and not nlp_contested

        example_labels = self._similar_example_co_labels(evidence)
        examples_contested = len(example_labels) > 1
        example_label = next(iter(example_labels)) if len(example_labels) == 1 else None

        if nlp_decisive and top is not None:
            co_label = top.co_id
            supports.append(
                f"Official course-outcome/syllabus relevance ranks {co_label} "
                f"highest (combined similarity {top.similarity:.3f})."
            )
            self._record_example_comparison(
                example_labels, co_label, conflicts, supports
            )
            if local_prediction is not None and local_prediction.co_label not in {
                "N/A",
                co_label,
            }:
                conflicts.append(
                    f"Existing local Qwen prediction selected "
                    f"{local_prediction.co_label}, while course-outcome/syllabus "
                    f"relevance selected {co_label}; the higher-priority "
                    "course evidence was retained."
                )
            self._record_co_analysis(analysis, co_label, supports, conflicts)
            self._record_co_sources(
                evidence, co_label, supports, conflicts, source_ids
            )
            return co_label, None

        if nlp_contested:
            conflicts.append(
                "Official course-outcome/syllabus relevance has multiple "
                "similarly ranked CO candidates."
            )

        if examples_contested:
            conflicts.append(
                "Similar faculty-approved examples have conflicting CO labels."
            )
        elif example_label is not None:
            supports.append(
                f"Similar faculty-approved examples consistently label "
                f"{example_label}."
            )
            if top is not None and top.co_id != example_label:
                conflicts.append(
                    f"The highest NLP candidate is {top.co_id}, while the "
                    f"faculty-approved examples support {example_label}; "
                    "the approved examples resolve the otherwise inconclusive "
                    "ranking."
                )
            if (
                local_prediction is not None
                and local_prediction.co_label in evidence.outcomes
                and local_prediction.co_label != example_label
            ):
                conflicts.append(
                    f"The existing local Qwen prediction selected "
                    f"{local_prediction.co_label}, while faculty-approved "
                    f"examples support {example_label}."
                )
            self._record_co_analysis(analysis, example_label, supports, conflicts)
            self._record_co_sources(
                evidence, example_label, supports, conflicts, source_ids
            )
            return example_label, None

        if nlp_contested or examples_contested:
            if (
                not examples_contested
                and local_prediction is not None
                and local_prediction.co_label in evidence.outcomes
            ):
                co_label = local_prediction.co_label
                supports.append(
                    f"Official course-context rankings are tied; the existing "
                    f"local Qwen prediction selected {co_label} and was retained."
                )
                self._record_co_analysis(analysis, co_label, supports, conflicts)
                self._record_co_sources(
                    evidence, co_label, supports, conflicts, source_ids
                )
                return co_label, None
            self._record_co_sources(
                evidence, None, supports, conflicts, source_ids
            )
            return (
                "N/A",
                "Official course-context rankings or faculty-approved examples "
                "conflict, and lower-priority evidence cannot safely break the tie.",
            )

        if analysis is not None and analysis.co_uncertain:
            conflicts.append(
                "Academic evidence analysis reports independent CO uncertainty."
            )
            self._record_co_sources(
                evidence, None, supports, conflicts, source_ids
            )
            return (
                "N/A",
                "The academic evidence analysis reports insufficient CO evidence.",
            )

        if (
            local_prediction is not None
            and local_prediction.co_label in evidence.outcomes
        ):
            co_label = local_prediction.co_label
            supports.append(
                f"The existing local Qwen prediction selected {co_label}; "
                "no higher-priority CO evidence provided a decisive alternative."
            )
            if top is not None and top.co_id != co_label:
                conflicts.append(
                    f"NLP ranks {top.co_id} above the existing local Qwen "
                    f"prediction {co_label}, but the ranking is not decisive."
                )
            self._record_co_analysis(analysis, co_label, supports, conflicts)
            self._record_co_sources(
                evidence, co_label, supports, conflicts, source_ids
            )
            return co_label, None

        self._record_co_sources(
            evidence, None, supports, conflicts, source_ids
        )
        return (
            "N/A",
            "No official course-context, approved-example, or local prediction "
            "evidence supports one available CO sufficiently.",
        )

    def _decide_bl(
        self,
        evidence: _PreparedEvidence,
        local_prediction: PredictionResult | None,
        analysis: AcademicEvidenceReport | None,
        supports: list[str],
        conflicts: list[str],
        source_ids: set[str],
    ) -> tuple[str, str | None]:
        analysis_levels = (
            set(analysis.possible_bl_levels)
            if analysis is not None
            else set()
        )
        analysis_decisive = (
            analysis is not None
            and not analysis.bl_uncertain
            and len(analysis_levels) == 1
            and next(iter(analysis_levels)) in _BL_LEVELS
        )
        approved_levels = self._similar_example_bl_levels(evidence)
        examples_contested = len(approved_levels) > 1
        example_level = (
            next(iter(approved_levels)) if len(approved_levels) == 1 else None
        )

        if analysis_decisive and analysis is not None:
            bl_label = next(iter(analysis_levels))
            supports.append(
                f"Academic analysis identifies {bl_label} from the question's "
                "cognitive demand."
            )
            if example_level is not None and example_level != bl_label:
                conflicts.append(
                    f"Faculty-approved examples support {example_level}, while "
                    f"academic analysis identifies {bl_label}; the higher-priority "
                    "cognitive-demand analysis was retained."
                )
            elif examples_contested:
                conflicts.append(
                    "Similar faculty-approved examples have conflicting BL labels."
                )
            if (
                local_prediction is not None
                and local_prediction.bl_label in _LEGACY_BL_LEVELS
                and local_prediction.bl_label != bl_label
            ):
                conflicts.append(
                    f"The existing local prediction returned "
                    f"{local_prediction.bl_label}, while academic analysis "
                    f"identified {bl_label}; the higher-priority cognitive-demand "
                    "analysis was retained."
                )
            self._record_bl_sources(analysis, supports, source_ids)
            return bl_label, None

        if analysis is not None and analysis_levels:
            if analysis.bl_uncertain or len(analysis_levels) != 1:
                conflicts.append(
                    "Academic analysis does not identify one certain revised "
                    "Bloom level."
                )

        if examples_contested:
            conflicts.append(
                "Similar faculty-approved examples have conflicting BL labels."
            )
        elif example_level is not None:
            possible_levels = analysis_levels - {"N/A"}
            if possible_levels and example_level not in possible_levels:
                conflicts.append(
                    f"Faculty-approved examples support {example_level}, outside "
                    "the possible levels reported by academic analysis."
                )
                self._record_bl_sources(analysis, supports, source_ids)
                return (
                    "N/A",
                    "The approved examples conflict with the possible cognitive "
                    "demand levels in the academic analysis.",
                )
            supports.append(
                f"Similar faculty-approved examples consistently support "
                f"{example_level}."
            )
            if (
                local_prediction is not None
                and local_prediction.bl_label in _LEGACY_BL_LEVELS
                and local_prediction.bl_label != example_level
            ):
                conflicts.append(
                    f"The existing local prediction returned "
                    f"{local_prediction.bl_label}, while faculty-approved "
                    f"examples support {example_level}."
                )
            self._record_bl_sources(analysis, supports, source_ids)
            return example_level, None

        if examples_contested:
            self._record_bl_sources(analysis, supports, source_ids)
            return (
                "N/A",
                "Similar faculty-approved examples disagree about the cognitive "
                "level.",
            )

        if analysis is not None and (
            analysis.bl_uncertain or analysis_levels
        ):
            self._record_bl_sources(analysis, supports, source_ids)
            return (
                "N/A",
                "The available cognitive-demand analysis is uncertain or spans "
                "multiple levels.",
            )

        if local_prediction is not None:
            legacy_level = local_prediction.bl_label
            if legacy_level in _LEGACY_BL_LEVELS:
                supports.append(
                    f"The existing local Qwen prediction supports {legacy_level}; "
                    "this shared level ID is used only as a lower-priority hint."
                )
                self._record_bl_sources(analysis, supports, source_ids)
                return legacy_level, None
            if legacy_level not in {"N/A", "BL0"}:
                conflicts.append(
                    f"The existing local prediction returned unsupported "
                    f"Bloom label {legacy_level}."
                )

        self._record_bl_sources(analysis, supports, source_ids)
        return (
            "N/A",
            "No sufficiently supported revised BL1-BL6 cognitive-demand label "
            "is available.",
        )

    def _prepare_evidence(
        self,
        outcomes: Mapping[str, str],
        units: Mapping[str, Sequence[str]] | Sequence[Mapping[str, object]],
        rankings: (
            SyllabusAwareQueryResult | Sequence[CourseOutcomeRanking] | None
        ),
        web_research: ResearchEngineResult | Sequence[ResearchSource] | None,
        examples: Sequence[SimilarApprovedExample] | None,
        analysis: AcademicEvidenceReport | None,
    ) -> _PreparedEvidence:
        validated_outcomes = self._validate_outcomes(outcomes)
        validated_units = self._validate_units(units)
        validated_rankings = self._validate_rankings(rankings, validated_outcomes)
        validated_examples = self._validate_examples(examples, validated_outcomes)
        research_sources = self._validate_research_sources(
            web_research, validated_outcomes
        )
        catalog = self._validate_analysis(analysis, validated_outcomes)
        return _PreparedEvidence(
            outcomes=validated_outcomes,
            units=validated_units,
            rankings=validated_rankings,
            examples=validated_examples,
            research_sources=research_sources,
            source_catalog=catalog,
            academic_analysis=analysis,
        )

    @staticmethod
    def _validate_outcomes(outcomes: Mapping[str, str]) -> dict[str, str]:
        if not isinstance(outcomes, Mapping) or not outcomes:
            raise ValueError("Provide at least one official course outcome.")
        validated: dict[str, str] = {}
        for co_id, description in outcomes.items():
            if (
                not isinstance(co_id, str)
                or not re.fullmatch(r"CO[1-5]", co_id)
                or not isinstance(description, str)
                or not description.strip()
            ):
                raise ValueError("Official course outcomes must have valid CO1-CO5 IDs and descriptions.")
            if co_id in validated:
                raise ValueError("Official course outcome IDs must be unique.")
            validated[co_id] = description.strip()
        return validated

    @classmethod
    def _validate_units(
        cls,
        units: Mapping[str, Sequence[str]]
        | Sequence[Mapping[str, object]],
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        prepared: list[tuple[str, tuple[str, ...]]] = []
        if isinstance(units, Mapping):
            entries = units.items()
            for label, topics in entries:
                if not isinstance(label, str) or not label.strip():
                    raise ValueError("Every syllabus unit needs a label.")
                prepared.append(
                    (label.strip(), cls._validated_topics(topics))
                )
            return tuple(prepared)
        if isinstance(units, (str, bytes)) or not isinstance(units, Sequence):
            raise ValueError("Syllabus units must be a mapping or sequence.")
        for unit in units:
            if not isinstance(unit, Mapping):
                raise ValueError("A syllabus unit has an invalid structure.")
            raw_label = unit.get("unit_title") or unit.get("unit_number")
            label = str(raw_label or "").strip()
            if not label:
                raise ValueError("Every syllabus unit needs a label.")
            prepared.append((label, cls._validated_topics(unit.get("topics", []))))
        return tuple(prepared)

    @staticmethod
    def _validated_topics(topics: object) -> tuple[str, ...]:
        if isinstance(topics, (str, bytes)) or not isinstance(topics, Sequence):
            raise ValueError("Syllabus topics must be a sequence.")
        if any(not isinstance(topic, str) or not topic.strip() for topic in topics):
            raise ValueError("Syllabus topics must be non-empty text.")
        return tuple(topic.strip() for topic in topics)

    @staticmethod
    def _validate_rankings(
        rankings: (
            SyllabusAwareQueryResult | Sequence[CourseOutcomeRanking] | None
        ),
        outcomes: Mapping[str, str],
    ) -> tuple[CourseOutcomeRanking, ...]:
        if rankings is None:
            return ()
        if isinstance(rankings, SyllabusAwareQueryResult):
            candidate_ids = rankings.plausible_co_candidates
            if any(not isinstance(co_id, str) for co_id in candidate_ids):
                raise ValueError("An NLP candidate CO ID is invalid.")
            if len(set(candidate_ids)) != len(candidate_ids):
                raise ValueError("NLP candidate rankings contain duplicate CO IDs.")
            if any(co_id not in outcomes for co_id in candidate_ids):
                raise ValueError(
                    "NLP candidate rankings include a CO outside the official outcomes."
                )
            rows = [
                row for row in rankings.co_rankings
                if row.co_id in candidate_ids
            ]
        else:
            rows = rankings
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise ValueError("NLP rankings must be a sequence.")
        checked: list[CourseOutcomeRanking] = []
        seen: set[str] = set()
        for ranking in rows:
            if not isinstance(ranking, CourseOutcomeRanking):
                raise ValueError("An NLP ranking has an invalid type.")
            if ranking.co_id not in outcomes:
                continue
            if ranking.co_id in seen:
                raise ValueError("NLP rankings contain duplicate CO IDs.")
            if ranking.description.strip() != outcomes[ranking.co_id]:
                continue
            scores = (
                ranking.similarity,
                ranking.tfidf_similarity,
                ranking.concept_overlap,
                ranking.topic_similarity,
            )
            if any(
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
                for score in scores
            ):
                raise ValueError("An NLP ranking contains an invalid similarity score.")
            seen.add(ranking.co_id)
            checked.append(ranking)
        return tuple(checked)

    def _validate_examples(
        self,
        examples: Sequence[SimilarApprovedExample] | None,
        outcomes: Mapping[str, str],
    ) -> tuple[SimilarApprovedExample, ...]:
        if examples is None:
            return ()
        if isinstance(examples, (str, bytes)) or not isinstance(
            examples, Sequence
        ):
            raise ValueError("Faculty-approved examples must be a sequence.")
        checked: list[SimilarApprovedExample] = []
        for example in examples:
            if not isinstance(example, SimilarApprovedExample):
                raise ValueError("A faculty-approved example has an invalid type.")
            if (
                not isinstance(example.similarity, (int, float))
                or isinstance(example.similarity, bool)
                or not math.isfinite(example.similarity)
                or not 0 <= example.similarity <= 1
            ):
                raise ValueError("An approved-example similarity is invalid.")
            if not isinstance(example.co_label, str) or not isinstance(
                example.bl_label, str
            ):
                raise ValueError("Approved-example labels must be text.")
            if example.bl_scheme not in {
                "custom_bl0_bl5",
                "revised_bl1_bl6",
            }:
                raise ValueError("An approved example has an invalid Bloom scheme.")
            if (
                example.co_label not in outcomes
                and example.co_label != "N/A"
            ):
                continue
            if example.bl_label not in _BL_LEVELS | {"BL0", "N/A"}:
                continue
            if (
                example.bl_scheme == "custom_bl0_bl5"
                and example.bl_label == "BL6"
            ) or (
                example.bl_scheme == "revised_bl1_bl6"
                and example.bl_label == "BL0"
            ):
                continue
            checked.append(example)
        return tuple(checked)

    @staticmethod
    def _validate_research_sources(
        research: ResearchEngineResult | Sequence[ResearchSource] | None,
        outcomes: Mapping[str, str],
    ) -> tuple[ResearchSource, ...]:
        if research is None:
            return ()
        sources = research.sources if isinstance(research, ResearchEngineResult) else research
        if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
            raise ValueError("Web research sources must be a sequence.")
        for source in sources:
            if not isinstance(source, ResearchSource):
                raise ValueError("A web research source has an invalid type.")
        return tuple(
            source
            for source in sources
            if (
                source.candidate_co_id in outcomes
                and isinstance(source.title, str)
                and source.title.strip()
                and isinstance(source.url, str)
                and source.url.strip()
            )
        )

    @staticmethod
    def _validate_analysis(
        analysis: AcademicEvidenceReport | None,
        outcomes: Mapping[str, str],
    ) -> tuple[EvidenceSourceReference, ...]:
        if analysis is None:
            return ()
        if not isinstance(analysis, AcademicEvidenceReport):
            raise ValueError("Academic analysis has an invalid type.")
        seen: set[str] = set()
        for source in analysis.source_catalog:
            if not isinstance(source, EvidenceSourceReference):
                raise ValueError("Academic analysis contains an invalid source.")
            if (
                not isinstance(source.source_id, str)
                or not source.source_id
                or source.source_id in seen
                or source.candidate_co_id not in outcomes
                or not source.title.strip()
                or not source.url.strip()
            ):
                raise ValueError("Academic analysis contains invalid source references.")
            seen.add(source.source_id)
        if any(
            not isinstance(source_id, str) or source_id not in seen
            for source_id in analysis.actual_source_ids
        ):
            raise ValueError("Academic analysis cites a source outside its catalog.")
        if any(
            item.co_id not in outcomes
            or item.description != outcomes[item.co_id]
            for item in analysis.candidate_co_analysis
        ) or any(item.co_id not in outcomes for item in analysis.co_evidence):
            raise ValueError("Academic analysis contains an invalid CO reference.")
        if len(
            {item.co_id for item in analysis.candidate_co_analysis}
        ) != len(analysis.candidate_co_analysis):
            raise ValueError("Academic analysis contains duplicate CO candidates.")
        catalog_by_id = {
            source.source_id: source for source in analysis.source_catalog
        }
        if any(
            not isinstance(item.source_id, str)
            or item.source_id not in seen
            or catalog_by_id[item.source_id].candidate_co_id != item.co_id
            or not isinstance(item.relationship, str)
            or item.relationship not in {"supports", "contradicts", "contextual"}
            for item in analysis.co_evidence
        ):
            raise ValueError("Academic analysis contains invalid CO evidence.")
        if any(source_id not in seen for source_id in analysis.bl_source_ids):
            raise ValueError("Academic analysis cites an unknown Bloom evidence source.")
        expected_citations = {
            item.source_id for item in analysis.co_evidence
        } | set(analysis.bl_source_ids)
        if expected_citations != set(analysis.actual_source_ids):
            raise ValueError("Academic analysis source IDs do not match its citations.")
        if (
            not isinstance(analysis.co_uncertain, bool)
            or not isinstance(analysis.bl_uncertain, bool)
            or not analysis.possible_bl_levels
            or any(
                not isinstance(level, str)
                or level not in _BL_LEVELS | {"N/A"}
                for level in analysis.possible_bl_levels
            )
            or (
                "N/A" in analysis.possible_bl_levels
                and len(analysis.possible_bl_levels) != 1
            )
            or (
                analysis.possible_bl_levels == ("N/A",)
                and not analysis.bl_uncertain
            )
            or len(set(analysis.possible_bl_levels))
            != len(analysis.possible_bl_levels)
        ):
            raise ValueError("Academic analysis contains invalid Bloom evidence.")
        return analysis.source_catalog

    @classmethod
    def _is_course_relevant(
        cls, question: str, evidence: _PreparedEvidence
    ) -> bool:
        question_terms = cls._meaningful_terms(question)
        if not question_terms:
            return False
        course_text = " ".join(
            [
                *evidence.outcomes.values(),
                *(topic for _, topics in evidence.units for topic in topics),
                *(label for label, _ in evidence.units),
            ]
        )
        return bool(question_terms.intersection(cls._meaningful_terms(course_text)))

    @staticmethod
    def _meaningful_terms(text: str) -> set[str]:
        return {
            token.casefold()
            for token in _TOKEN_PATTERN.findall(text)
            if len(token) > 1 and token.casefold() not in _STOP_WORDS
        }

    def _similar_example_co_labels(
        self, evidence: _PreparedEvidence
    ) -> set[str]:
        return {
            example.co_label
            for example in evidence.examples
            if example.similarity >= self.min_example_similarity
            and example.co_label in evidence.outcomes
        }

    def _similar_example_bl_levels(
        self, evidence: _PreparedEvidence
    ) -> set[str]:
        return {
            example.bl_label
            for example in evidence.examples
            if example.similarity >= self.min_example_similarity
            and example.bl_label in _BL_LEVELS
            and getattr(example, "bl_scheme", "custom_bl0_bl5")
            == "revised_bl1_bl6"
        }

    @staticmethod
    def _record_example_comparison(
        example_labels: set[str],
        selected_co: str,
        conflicts: list[str],
        supports: list[str],
    ) -> None:
        if not example_labels:
            return
        if selected_co in example_labels:
            supports.append(
                f"Faculty-approved examples also support {selected_co}."
            )
        else:
            conflicts.append(
                f"Faculty-approved examples support {', '.join(sorted(example_labels))}, "
                f"not the higher-priority course-context result {selected_co}."
            )

    @staticmethod
    def _record_co_analysis(
        analysis: AcademicEvidenceReport | None,
        selected_co: str,
        supports: list[str],
        conflicts: list[str],
    ) -> None:
        if analysis is None:
            return
        for item in analysis.candidate_co_analysis:
            if item.co_id == selected_co:
                supports.append(
                    f"Academic analysis for {selected_co}: {item.analysis}"
                )
        for item in analysis.co_evidence:
            if item.co_id != selected_co:
                continue
            line = (
                f"Academic source {item.source_id} "
                f"{item.relationship} {selected_co}: {item.rationale}"
            )
            (conflicts if item.relationship == "contradicts" else supports).append(line)

    @staticmethod
    def _record_co_sources(
        evidence: _PreparedEvidence,
        selected_co: str | None,
        supports: list[str],
        conflicts: list[str],
        source_ids: set[str],
    ) -> None:
        analysis = evidence.academic_analysis
        catalog = {source.source_id: source for source in evidence.source_catalog}
        if analysis is not None:
            for item in analysis.co_evidence:
                if selected_co is not None and item.co_id != selected_co:
                    continue
                if item.source_id not in catalog:
                    continue
                source_ids.add(item.source_id)
                message = (
                    f"Academic source {item.source_id} "
                    f"({catalog[item.source_id].title}) {item.relationship} "
                    f"{item.co_id}; it is supporting technical context, not an "
                    "institutional CO decision."
                )
                (
                    conflicts
                    if item.relationship == "contradicts"
                    else supports
                ).append(message)

        cited_sources = {
            (source.url.casefold(), source.candidate_co_id)
            for source in evidence.source_catalog
        }
        for index, source in enumerate(evidence.research_sources, start=1):
            if selected_co is not None and source.candidate_co_id != selected_co:
                continue
            if (source.url.casefold(), source.candidate_co_id) in cited_sources:
                continue
            web_id = f"WEB-{index:04d}"
            source_ids.add(web_id)
            supports.append(
                f"Web source {web_id} ({source.title}) is technical context "
                f"associated with {source.candidate_co_id}; it was not used "
                "to assign an institutional CO."
            )

    @staticmethod
    def _record_bl_sources(
        analysis: AcademicEvidenceReport | None,
        supports: list[str],
        source_ids: set[str],
    ) -> None:
        if analysis is None:
            return
        source_catalog = {source.source_id: source for source in analysis.source_catalog}
        for source_id in analysis.bl_source_ids:
            if source_id not in source_catalog:
                continue
            source_ids.add(source_id)
            supports.append(
                f"Academic source {source_id} ({source_catalog[source_id].title}) "
                "was cited as Bloom-level evidence."
            )

    @staticmethod
    def _sources_used(
        evidence: _PreparedEvidence,
        source_ids: set[str],
    ) -> tuple[DecisionSource, ...]:
        sources: list[DecisionSource] = []
        for source in evidence.source_catalog:
            if source.source_id not in source_ids:
                continue
            matched_research = next(
                (
                    item
                    for item in evidence.research_sources
                    if item.url.casefold() == source.url.casefold()
                    and item.candidate_co_id == source.candidate_co_id
                ),
                None,
            )
            sources.append(
                DecisionSource(
                    source_id=source.source_id,
                    title=source.title,
                    url=source.url,
                    candidate_co_id=source.candidate_co_id,
                    search_query=(
                        matched_research.search_query
                        if matched_research is not None
                        else None
                    ),
                    search_snippet=source.search_snippet,
                    extracted_content=source.extracted_content,
                    publication_date=source.publication_date,
                    retrieved_at=source.retrieved_at,
                    search_provider=source.search_provider,
                    extraction_status=source.extraction_status,
                    source_quality=source.source_quality,
                )
            )
        for index, source in enumerate(evidence.research_sources, start=1):
            source_id = f"WEB-{index:04d}"
            if source_id not in source_ids:
                continue
            sources.append(
                DecisionSource(
                    source_id=source_id,
                    title=source.title,
                    url=source.url,
                    candidate_co_id=source.candidate_co_id,
                    search_query=source.search_query,
                    search_snippet=source.search_snippet,
                    extracted_content=source.extracted_content,
                    publication_date=source.publication_date,
                    retrieved_at=source.retrieved_at,
                    search_provider=source.search_provider,
                    extraction_status=source.extraction_status,
                    source_quality=source.source_quality,
                )
            )
        return tuple(sources)

    @staticmethod
    def _validate_local_prediction(
        prediction: PredictionResult | None,
        outcomes: Mapping[str, str],
    ) -> None:
        if prediction is None:
            return
        if not isinstance(prediction, PredictionResult):
            raise ValueError("The existing local prediction has an invalid type.")
        if prediction.co_label not in {*outcomes, "N/A"}:
            raise ValueError("The existing prediction has an unavailable CO label.")
        if prediction.bl_label not in {*_LEGACY_BL_LEVELS, "BL0", "N/A"}:
            raise ValueError("The existing prediction has an invalid custom BL label.")

    @staticmethod
    def _required_text(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be non-empty text.")
        return value.strip()
