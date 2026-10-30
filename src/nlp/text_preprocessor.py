import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_MIN_SIMILARITY = 0.15
DEFAULT_MAX_RESULTS = 3


def preprocess_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


@dataclass(frozen=True)
class SimilarApprovedExample:
    question_text: str
    co_label: str
    bl_label: str
    similarity: float


class ApprovedExampleRetriever:
    """Retrieve current faculty-reviewed examples for one course at a time."""

    def __init__(
        self,
        database: Any,
        *,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("Minimum similarity must be between 0 and 1.")
        if max_results < 1:
            raise ValueError("Maximum results must be at least one.")
        self.database = database
        self.min_similarity = min_similarity
        self.max_results = max_results

    def retrieve(
        self, course_code: str, question_text: str
    ) -> list[SimilarApprovedExample]:
        query = preprocess_text(question_text)
        if not query:
            return []

        reviewed_records = self.database.list_question_history(
            course_code=course_code
        )
        approved_records = [
            record
            for record in reviewed_records
            if record["label_review_status"] in {"approved", "corrected"}
            and record["approved_co"] is not None
            and record["approved_bl"] is not None
        ]
        if not approved_records:
            return []

        texts = [
            preprocess_text(record["question_text"])
            for record in approved_records
        ]
        try:
            vectorizer = TfidfVectorizer(
                lowercase=False,
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            vectors = vectorizer.fit_transform([*texts, query])
        except ValueError:
            return []

        scores = cosine_similarity(vectors[-1], vectors[:-1]).ravel()
        ranked = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return [
            SimilarApprovedExample(
                question_text=approved_records[index]["question_text"],
                co_label=approved_records[index]["approved_co"],
                bl_label=approved_records[index]["approved_bl"],
                similarity=float(score),
            )
            for index, score in ranked
            if score >= self.min_similarity
        ][: self.max_results]
