from src.nlp.text_preprocessor import SimilarApprovedExample
from src.prediction.hybrid_prediction_engine import HybridPredictionEngine
from src.prediction.predictor import PredictionResult
from src.research.academic_evidence_analyzer import (
    AcademicEvidenceReport,
    CandidateCOAnalysis,
    COEvidence,
    EvidenceSourceReference,
)
from src.research.research_engine import ResearchSource
from src.research.syllabus_query_generator import CourseOutcomeRanking


OUTCOMES = {
    "CO1": "Analyze network routing algorithms and shortest paths.",
    "CO2": "Apply data link framing and error detection.",
}
UNITS = {
    "Unit I: Routing": ["Dijkstra algorithm", "Shortest paths"],
    "Unit II: Data link": ["Framing", "Error detection"],
}
QUESTION = "Analyze Dijkstra routing algorithms for shortest paths."


def _ranking(co_id, score, description=None, topic_score=0.5):
    return CourseOutcomeRanking(
        co_id=co_id,
        description=description or OUTCOMES[co_id],
        similarity=score,
        tfidf_similarity=score,
        concept_overlap=score,
        topic_similarity=topic_score,
        matched_topics=(),
    )


def _prediction(co="CO1", bl="BL3"):
    return PredictionResult(
        co_label=co,
        bl_label=bl,
        status="predicted",
        question_id="question-id",
        prediction_id="prediction-id",
        syllabus_version_id=None,
        model_name="local-test-model",
        model_version="test",
    )


def _analysis(levels=("BL3",), *, co_uncertain=False, bl_uncertain=False):
    source = EvidenceSourceReference(
        source_id="SRC-0001",
        candidate_co_id="CO1",
        title="Routing Documentation",
        url="https://example.edu/routing",
        search_snippet="Dijkstra computes shortest paths.",
        extracted_content="Dijkstra's algorithm finds shortest paths.",
        publication_date=None,
        retrieved_at="2026-11-02T00:00:00+00:00",
        search_provider="DDGS",
        extraction_status="success",
        source_quality="university",
    )
    return AcademicEvidenceReport(
        extracted_concepts=("Dijkstra", "routing", "shortest paths"),
        cognitive_action="Analyze",
        candidate_co_analysis=(
            CandidateCOAnalysis(
                co_id="CO1",
                description=OUTCOMES["CO1"],
                question_objective="Analyze a routing algorithm.",
                syllabus_relationship="Matches the routing unit.",
                technical_concept_relevance="Dijkstra is a routing concept.",
                learning_outcome_relevance="Matches CO1.",
                analysis="Evidence supports the routing outcome.",
            ),
        ),
        co_evidence=(
            COEvidence(
                co_id="CO1",
                source_id="SRC-0001",
                relationship="supports",
                rationale="The source describes shortest-path routing.",
            ),
        ),
        possible_bl_levels=levels,
        bl_reasoning="The task requires examining an algorithm.",
        bl_source_ids=("SRC-0001",) if levels != ("N/A",) else (),
        actual_source_ids=("SRC-0001",),
        source_catalog=(source,),
        limitations=(),
        co_uncertain=co_uncertain,
        bl_uncertain=bl_uncertain,
    )


def _example(
    co="CO1",
    bl="BL3",
    similarity=0.8,
    bl_scheme="custom_bl0_bl5",
):
    return SimilarApprovedExample(
        question_text="Analyze Dijkstra routing.",
        co_label=co,
        bl_label=bl,
        similarity=similarity,
        bl_scheme=bl_scheme,
    )


def _web_source(candidate_co_id="CO1"):
    return ResearchSource(
        candidate_co_id=candidate_co_id,
        search_query="Dijkstra routing",
        title="Routing Technical Notes",
        url="https://example.edu/routing-notes",
        search_snippet="Dijkstra computes shortest paths.",
        extracted_content=None,
        publication_date=None,
        retrieved_at="2026-11-02T00:00:00+00:00",
        search_provider="DDGS",
        extraction_status="skipped",
        relevance_score=0.8,
        source_quality="university",
    )


def test_hybrid_engine_combines_agreeing_evidence_and_formats_two_lines():
    result = HybridPredictionEngine().predict(
        QUESTION,
        OUTCOMES,
        UNITS,
        local_prediction=_prediction(),
        nlp_candidate_co_rankings=[_ranking("CO1", 0.72), _ranking("CO2", 0.12)],
        academic_analysis=_analysis(),
        faculty_approved_examples=[_example()],
    )

    assert result.co_label == "CO1"
    assert result.bl_label == "BL3"
    assert result.status == "predicted"
    assert result.format_output() == "CO level - CO1\nBL level - BL3"
    assert result.decision_record.pipeline_mode == "hybrid"
    assert result.decision_record.sources_used[0].source_id == "SRC-0001"
    assert not result.decision_record.conflicting_evidence


def test_official_course_context_takes_priority_and_records_disagreement():
    result = HybridPredictionEngine().predict(
        QUESTION,
        OUTCOMES,
        UNITS,
        local_prediction=_prediction(co="CO2"),
        nlp_candidate_co_rankings=[_ranking("CO1", 0.72), _ranking("CO2", 0.12)],
        faculty_approved_examples=[_example(co="CO2")],
    )

    assert result.co_label == "CO1"
    assert result.decision_record.conflicting_evidence
    assert any("higher-priority" in item for item in result.decision_record.conflicting_evidence)


def test_tied_multiple_co_candidates_fall_back_to_valid_qwen_prediction():
    result = HybridPredictionEngine().predict(
        QUESTION,
        OUTCOMES,
        UNITS,
        local_prediction=_prediction(co="CO1"),
        nlp_candidate_co_rankings=[_ranking("CO1", 0.62), _ranking("CO2", 0.60)],
    )

    assert result.co_label == "CO1"
    assert result.decision_record.co_abstention_reason is None
    assert any("local Qwen prediction selected CO1" in item for item in result.decision_record.supporting_evidence)
    assert any("similarly ranked" in item for item in result.decision_record.conflicting_evidence)


def test_unrelated_question_returns_independent_na_with_invalid_status():
    result = HybridPredictionEngine().predict(
        "Hello, how are you?",
        OUTCOMES,
        UNITS,
        local_prediction=_prediction(),
    )

    assert result.status == "invalid_question"
    assert result.format_output() == "CO level - N/A\nBL level - N/A"


def test_missing_web_results_do_not_prevent_local_evidence_decisions():
    result = HybridPredictionEngine().predict(
        QUESTION,
        OUTCOMES,
        UNITS,
        local_prediction=_prediction(),
        nlp_candidate_co_rankings=[_ranking("CO1", 0.72)],
        web_research=[],
        faculty_approved_examples=[_example()],
    )

    assert result.co_label == "CO1"
    assert result.bl_label == "BL3"
    assert result.decision_record.pipeline_mode == "local-only"
    assert not result.decision_record.sources_used


def test_co_and_bl_uncertainty_are_independent():
    result = HybridPredictionEngine().predict(
        QUESTION,
        OUTCOMES,
        UNITS,
        local_prediction=_prediction(),
        nlp_candidate_co_rankings=[_ranking("CO1", 0.72)],
        academic_analysis=_analysis(
            levels=("N/A",), bl_uncertain=True
        ),
    )

    assert result.co_label == "CO1"
    assert result.bl_label == "N/A"
    assert result.status == "insufficient_context"
    assert result.decision_record.co_abstention_reason is None
    assert result.decision_record.bl_abstention_reason


def test_approved_examples_can_support_bl_when_analysis_abstains():
    result = HybridPredictionEngine().predict(
        QUESTION,
        OUTCOMES,
        UNITS,
        academic_analysis=_analysis(levels=("N/A",), bl_uncertain=True),
        faculty_approved_examples=[
            _example(bl="BL4", bl_scheme="revised_bl1_bl6")
        ],
    )

    assert result.bl_label == "BL4"
    assert result.decision_record.bl_abstention_reason is None


def test_co_can_abstain_while_bl_is_supported():
    result = HybridPredictionEngine().predict(
        QUESTION,
        OUTCOMES,
        UNITS,
        local_prediction=_prediction(co="N/A", bl="BL3"),
        nlp_candidate_co_rankings=[
            _ranking("CO1", 0.62),
            _ranking("CO2", 0.60),
        ],
    )

    assert result.co_label == "N/A"
    assert result.bl_label == "BL3"
    assert result.decision_record.co_abstention_reason
    assert result.decision_record.bl_abstention_reason is None


def test_web_sources_are_recorded_as_context_but_never_assign_labels():
    result = HybridPredictionEngine().predict(
        QUESTION,
        OUTCOMES,
        UNITS,
        web_research=[_web_source()],
    )

    assert result.co_label == "N/A"
    assert result.bl_label == "N/A"
    assert result.decision_record.sources_used[0].source_id == "WEB-0001"
    assert any(
        "not used to assign an institutional CO" in item
        for item in result.decision_record.supporting_evidence
    )


def test_revised_bl6_can_only_be_selected_from_the_new_analysis():
    result = HybridPredictionEngine().predict(
        QUESTION,
        OUTCOMES,
        UNITS,
        academic_analysis=_analysis(levels=("BL6",)),
        nlp_candidate_co_rankings=[_ranking("CO1", 0.72)],
    )

    assert result.bl_label == "BL6"


def test_local_only_fallback_uses_legacy_shared_level_without_changing_predictor():
    result = HybridPredictionEngine().predict(
        QUESTION,
        OUTCOMES,
        UNITS,
        local_prediction=_prediction(co="CO1", bl="BL5"),
    )

    assert result.co_label == "CO1"
    assert result.bl_label == "BL5"
    assert result.decision_record.pipeline_mode == "local-only"
