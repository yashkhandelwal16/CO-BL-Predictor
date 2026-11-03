import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.nlp.text_preprocessor import (
    normalize_academic_term,
    preprocess_text,
    tfidf_similarity_scores,
)


DEFAULT_MAX_CANDIDATES = 3
DEFAULT_MIN_CO_SIMILARITY = 0.08
DEFAULT_MIN_TOPIC_SIMILARITY = 0.05
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[+#][a-z0-9+#]*|(?:\.[a-z0-9]+)+)?")
_QUESTION_PREFIX_PATTERN = re.compile(
    r"^\s*(?:(?:question|q)\s*\d*\s*[:.)-]\s*|please\s+)",
    re.IGNORECASE,
)
_STOP_WORDS = frozenset(
    {
        "a", "about", "after", "all", "an", "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "between", "both", "but",
        "by", "can", "could", "describe", "determine", "discuss", "do", "does",
        "during", "each", "evaluate", "explain", "for", "from", "give", "has",
        "have", "how", "identify", "if", "in", "into", "is", "it", "its",
        "may", "implement", "might", "must", "of", "on", "or", "should",
        "show", "solve", "state",
        "that", "the", "their", "then", "there", "these", "this", "those",
        "through", "to", "under", "use", "using", "was", "were", "what",
        "when", "where", "which", "who", "why", "will", "with", "would",
        "write", "analyze", "apply", "calculate", "classify", "compare",
        "compute", "create", "design", "derive", "develop", "formulate",
        "construct", "contrast", "demonstrate", "differentiate", "examine",
        "illustrate", "justify", "list", "recall", "summarize", "verify",
    }
)
_GENERIC_BRIDGE_TERMS = frozenset(
    {
        "approach",
        "approaches",
        "course",
        "method",
        "methods",
        "program",
        "programs",
        "system",
        "systems",
        "technique",
        "techniques",
        "topic",
        "topics",
    }
)


@dataclass(frozen=True)
class SyllabusTopicMatch:
    unit: str
    topic: str
    similarity: float


@dataclass(frozen=True)
class CourseOutcomeRanking:
    co_id: str
    description: str
    similarity: float
    tfidf_similarity: float
    concept_overlap: float
    topic_similarity: float
    matched_topics: tuple[SyllabusTopicMatch, ...]


@dataclass(frozen=True)
class GeneratedSearchQuery:
    co_id: str
    query: str
    reason: str


@dataclass(frozen=True)
class SyllabusAwareQueryResult:
    extracted_concepts: tuple[str, ...]
    technical_terminology: tuple[str, ...]
    subject_area: str
    cognitive_action: str
    actual_cognitive_task: str
    relevant_syllabus_topics: tuple[SyllabusTopicMatch, ...]
    co_rankings: tuple[CourseOutcomeRanking, ...]
    plausible_co_candidates: tuple[str, ...]
    generated_queries: tuple[GeneratedSearchQuery, ...]


class SyllabusAwareQueryGenerator:
    """Rank supplied course outcomes and generate searches for plausible ones."""

    def __init__(
        self,
        *,
        max_candidates: int = DEFAULT_MAX_CANDIDATES,
        min_co_similarity: float = DEFAULT_MIN_CO_SIMILARITY,
        min_topic_similarity: float = DEFAULT_MIN_TOPIC_SIMILARITY,
    ) -> None:
        if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
            raise ValueError("Maximum candidate count must be a positive integer.")
        if max_candidates < 1:
            raise ValueError("Maximum candidate count must be a positive integer.")
        self.max_candidates = max_candidates
        self.min_co_similarity = self._validate_threshold(
            min_co_similarity, "CO similarity"
        )
        self.min_topic_similarity = self._validate_threshold(
            min_topic_similarity, "Topic similarity"
        )

    def generate(
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
    ) -> SyllabusAwareQueryResult:
        """Analyze a question against only the supplied course information."""
        question = self._required_text(academic_question, "Academic question")
        name = self._required_text(course_name, "Course name")
        code = self._required_text(course_code, "Course code")
        action = self._required_text(
            detected_cognitive_action, "Detected cognitive action"
        )
        outcomes = self._validate_outcomes(course_outcomes)
        topics = self._flatten_topics(unit_topics)
        concepts = self._validate_concepts(relevant_technical_concepts)

        question_terms = self._meaningful_terms(question)
        evidence_terms = question_terms | {
            term
            for concept in concepts
            for term in self._meaningful_terms(concept)
        }
        topic_query = " ".join((question, *concepts))
        topic_texts = [topic.topic for topic in topics]
        topic_tfidf_scores = tfidf_similarity_scores(topic_query, topic_texts)
        topic_matches: list[SyllabusTopicMatch] = []
        for topic, tfidf_score in zip(topics, topic_tfidf_scores, strict=True):
            overlap = self._overlap(evidence_terms, self._meaningful_terms(topic.topic))
            score = 0.8 * tfidf_score + 0.2 * overlap
            if score >= self.min_topic_similarity:
                topic_matches.append(
                    SyllabusTopicMatch(topic.unit, topic.topic, score)
                )
        topic_matches.sort(key=lambda match: (-match.similarity, match.unit, match.topic))

        descriptions = [description for _, description in outcomes]
        direct_scores = tfidf_similarity_scores(question, descriptions)
        outcome_term_sets = [
            self._meaningful_terms(description) for _, description in outcomes
        ]
        topic_term_sets = [
            self._meaningful_terms(topic.topic) - _GENERIC_BRIDGE_TERMS
            for topic in topics
        ]
        outcome_bridge_terms = [
            terms - _GENERIC_BRIDGE_TERMS for terms in outcome_term_sets
        ]

        rankings: list[CourseOutcomeRanking] = []
        for index, (co_id, description) in enumerate(outcomes):
            concept_overlap = self._overlap(evidence_terms, outcome_term_sets[index])
            matching_topics: list[SyllabusTopicMatch] = []
            bridged_scores: list[float] = []
            for topic_index, topic in enumerate(topics):
                topic_score = self._topic_score_for(topic, topic_matches)
                if topic_score:
                    topic_outcome_overlap = self._overlap(
                        topic_term_sets[topic_index],
                        outcome_bridge_terms[index],
                    )
                    bridged_score = (
                        topic_score * topic_outcome_overlap
                    )
                    if bridged_score:
                        bridged_scores.append(bridged_score)
                        matching_topics.append(
                            SyllabusTopicMatch(
                                topic.unit,
                                topic.topic,
                                bridged_score,
                            )
                        )

            topic_similarity = max(bridged_scores, default=0.0)
            tfidf_score = direct_scores[index]
            combined_score = (
                0.65 * tfidf_score
                + 0.20 * topic_similarity
                + 0.15 * concept_overlap
            )
            matching_topics.sort(
                key=lambda match: (-match.similarity, match.unit, match.topic)
            )
            rankings.append(
                CourseOutcomeRanking(
                    co_id=co_id,
                    description=description,
                    similarity=combined_score,
                    tfidf_similarity=tfidf_score,
                    concept_overlap=concept_overlap,
                    topic_similarity=topic_similarity,
                    matched_topics=tuple(matching_topics),
                )
            )

        rankings.sort(key=lambda ranking: -ranking.similarity)
        plausible = self._select_plausible_candidates(rankings)
        ranking_by_id = {ranking.co_id: ranking for ranking in rankings}
        question_concepts = self._extract_question_concepts(
            question,
            [
                *concepts,
                *(topic.topic for topic in topics),
                *(description for _, description in outcomes),
            ],
        )
        terminology = self._extract_question_concepts(
            question,
            concepts,
        )
        queries = tuple(
            self._make_query(
                question=question,
                course_name=name,
                course_code=code,
                cognitive_action=action,
                concepts=concepts,
                topic_matches=ranking_by_id[co_id].matched_topics,
                ranking=ranking_by_id[co_id],
            )
            for co_id in plausible
        )
        return SyllabusAwareQueryResult(
            extracted_concepts=tuple(
                self._unique_texts((*concepts, *question_concepts))
            ),
            technical_terminology=tuple(terminology),
            subject_area=name,
            cognitive_action=action,
            actual_cognitive_task=self._task_text(question),
            relevant_syllabus_topics=tuple(topic_matches),
            co_rankings=tuple(rankings),
            plausible_co_candidates=plausible,
            generated_queries=queries,
        )

    def _select_plausible_candidates(
        self, rankings: Sequence[CourseOutcomeRanking]
    ) -> tuple[str, ...]:
        if not rankings or rankings[0].similarity < self.min_co_similarity:
            return ()
        relative_threshold = max(
            self.min_co_similarity,
            rankings[0].similarity * 0.4,
        )
        return tuple(
            ranking.co_id
            for ranking in rankings
            if ranking.similarity >= relative_threshold
        )[: self.max_candidates]

    @staticmethod
    def _make_query(
        *,
        question: str,
        course_name: str,
        course_code: str,
        cognitive_action: str,
        concepts: Sequence[str],
        topic_matches: Sequence[SyllabusTopicMatch],
        ranking: CourseOutcomeRanking,
    ) -> GeneratedSearchQuery:
        related_topics = [match.topic for match in topic_matches[:3]]
        query_parts = [
            course_name,
            course_code,
            question,
            ranking.description,
            *concepts,
            *related_topics,
            cognitive_action,
            "academic learning objectives",
        ]
        query = " ".join(SyllabusAwareQueryGenerator._unique_texts(query_parts))
        evidence = [
            f"the question-to-CO TF-IDF score is {ranking.tfidf_similarity:.2f}"
        ]
        if concepts:
            evidence.append("relevant technical concepts were supplied")
        if related_topics:
            evidence.append(
                "related syllabus topics include "
                + ", ".join(f'"{topic}"' for topic in related_topics)
            )
        reason = (
            f"Generated for {ranking.co_id} because "
            + "; ".join(evidence)
            + ". The query combines the question, this official CO description, "
            "course identity, and supported learning context; web results do not "
            "assign or confirm the CO."
        )
        return GeneratedSearchQuery(ranking.co_id, query, reason)

    @staticmethod
    def _flatten_topics(
        unit_topics: (
            Mapping[str, Sequence[str]]
            | Sequence[Mapping[str, object]]
        ),
    ) -> list[SyllabusTopicMatch]:
        flattened: list[SyllabusTopicMatch] = []
        if isinstance(unit_topics, Mapping):
            unit_entries = unit_topics.items()
            for unit, topics in unit_entries:
                label = SyllabusAwareQueryGenerator._required_text(
                    unit, "Unit label"
                )
                if isinstance(topics, (str, bytes)) or not isinstance(
                    topics, Sequence
                ):
                    raise ValueError("Each unit must contain a sequence of topics.")
                for topic in topics:
                    flattened.append(
                        SyllabusTopicMatch(
                            label,
                            SyllabusAwareQueryGenerator._required_text(
                                topic, "Syllabus topic"
                            ),
                            0.0,
                        )
                    )
            return flattened

        if isinstance(unit_topics, (str, bytes)) or not isinstance(
            unit_topics, Sequence
        ):
            raise ValueError("Unit-wise syllabus topics must be a mapping or sequence.")
        for unit in unit_topics:
            if not isinstance(unit, Mapping):
                raise ValueError("Each syllabus unit must be a mapping.")
            label = unit.get("unit_title") or unit.get("unit_number")
            unit_name = SyllabusAwareQueryGenerator._required_text(
                str(label) if label is not None else "", "Unit label"
            )
            unit_values = unit.get("topics", [])
            if isinstance(unit_values, (str, bytes)) or not isinstance(
                unit_values, Sequence
            ):
                raise ValueError("Each unit must contain a sequence of topics.")
            for topic in unit_values:
                flattened.append(
                    SyllabusTopicMatch(
                        unit_name,
                        SyllabusAwareQueryGenerator._required_text(
                            topic, "Syllabus topic"
                        ),
                        0.0,
                    )
                )
        return flattened

    @staticmethod
    def _validate_outcomes(
        course_outcomes: Mapping[str, str],
    ) -> list[tuple[str, str]]:
        if not isinstance(course_outcomes, Mapping) or not course_outcomes:
            raise ValueError("Provide at least one official course outcome.")
        validated: list[tuple[str, str]] = []
        seen: set[str] = set()
        for co_id, description in course_outcomes.items():
            if not isinstance(co_id, str) or not co_id.strip():
                raise ValueError("Every official course outcome needs an ID.")
            if not isinstance(description, str) or not description.strip():
                raise ValueError("Every official course outcome needs a description.")
            normalized_id = co_id.strip()
            if normalized_id.casefold() in seen:
                raise ValueError("Official course outcome IDs must be unique.")
            seen.add(normalized_id.casefold())
            validated.append((normalized_id, description))
        return validated

    @staticmethod
    def _validate_concepts(concepts: Sequence[str]) -> tuple[str, ...]:
        if isinstance(concepts, (str, bytes)) or not isinstance(concepts, Sequence):
            raise ValueError("Relevant technical concepts must be provided as a list.")
        return tuple(
            SyllabusAwareQueryGenerator._required_text(concept, "Technical concept")
            for concept in concepts
        )

    @staticmethod
    def _required_text(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} is required.")
        return value.strip()

    @staticmethod
    def _meaningful_terms(text: str) -> set[str]:
        return {
            normalize_academic_term(token)
            for token in _TOKEN_PATTERN.findall(preprocess_text(text))
            if token not in _STOP_WORDS and len(token) > 1
        }

    @staticmethod
    def _overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left.intersection(right)) / len(left)

    @staticmethod
    def _topic_score_for(
        topic: SyllabusTopicMatch,
        matches: Sequence[SyllabusTopicMatch],
    ) -> float:
        for match in matches:
            if match.unit == topic.unit and match.topic == topic.topic:
                return match.similarity
        return 0.0

    @classmethod
    def _extract_question_concepts(
        cls, question: str, sources: Sequence[str]
    ) -> list[str]:
        question_tokens = _TOKEN_PATTERN.findall(preprocess_text(question))
        normalized_question = preprocess_text(question)
        if not question_tokens:
            return []
        question_ngrams = {
            " ".join(question_tokens[index : index + size])
            for size in (1, 2, 3)
            for index in range(len(question_tokens) - size + 1)
            if all(
                token not in _STOP_WORDS and len(token) > 1
                for token in question_tokens[index : index + size]
            )
        }
        source_ngrams: dict[str, str] = {}
        for source in sources:
            source_tokens = _TOKEN_PATTERN.findall(preprocess_text(source))
            for size in (1, 2, 3):
                for index in range(len(source_tokens) - size + 1):
                    phrase_tokens = source_tokens[index : index + size]
                    if any(
                        token in _STOP_WORDS or len(token) <= 1
                        for token in phrase_tokens
                    ):
                        continue
                    phrase = " ".join(phrase_tokens)
                    source_ngrams.setdefault(phrase, phrase)
        matched = question_ngrams.intersection(source_ngrams)
        return sorted(
            (source_ngrams[phrase] for phrase in matched),
            key=lambda phrase: (
                -len(phrase.split()),
                normalized_question.find(phrase),
                phrase,
            ),
        )

    @staticmethod
    def _unique_texts(values: Sequence[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = value.strip()
            normalized = preprocess_text(cleaned)
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(cleaned)
        return unique

    @staticmethod
    def _task_text(question: str) -> str:
        task = _QUESTION_PREFIX_PATTERN.sub("", question).strip()
        return task.rstrip(" ?!.")

    @staticmethod
    def _validate_threshold(value: float, label: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} threshold must be between 0 and 1.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{label} threshold must be between 0 and 1.")
        return float(value)
