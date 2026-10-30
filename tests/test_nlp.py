from src.courses.course_manager import COURSE_OUTCOME_IDS, CourseManager
from src.database.db_manager import DatabaseManager
from src.nlp.text_preprocessor import ApprovedExampleRetriever


def _outcomes() -> dict[str, str]:
    return {
        outcome_id: f"Students will demonstrate {outcome_id}."
        for outcome_id in COURSE_OUTCOME_IDS
    }


def _save_reviewed_example(
    database: DatabaseManager,
    course_code: str,
    question_text: str,
    co_label: str,
    bl_label: str,
    *,
    review_status: str = "approved",
) -> str:
    question = database.create_question(course_code, question_text)
    prediction = database.create_prediction(
        question["id"], "CO1", "BL1"
    )
    database.save_approved_labels(
        question["id"],
        co_label,
        bl_label,
        prediction_id=prediction["id"],
        review_status=review_status,
    )
    return question["id"]


def test_retriever_uses_only_similar_approved_examples_for_selected_course():
    database = DatabaseManager()
    manager = CourseManager(database)
    manager.add_course("Computer Networks", "CS305", _outcomes())
    manager.add_course("Biology", "BIO201", _outcomes())
    _save_reviewed_example(
        database,
        "CS305",
        "Explain how a router forwards packets with a routing table.",
        "CO3",
        "BL2",
    )
    _save_reviewed_example(
        database,
        "CS305",
        "Describe the operation of an Ethernet switch.",
        "CO2",
        "BL3",
    )
    _save_reviewed_example(
        database,
        "BIO201",
        "Explain how a router forwards packets with a routing table.",
        "CO4",
        "BL4",
    )
    unreviewed_question = database.create_question(
        "CS305", "Explain how a router forwards packets."
    )
    database.create_prediction(
        unreviewed_question["id"], "CO4", "BL4"
    )

    retrieved = ApprovedExampleRetriever(database).retrieve(
        "CS305", "Describe how a router forwards packets using a routing table."
    )

    assert retrieved
    assert retrieved[0].co_label == "CO3"
    assert retrieved[0].bl_label == "BL2"
    assert {example.co_label for example in retrieved} <= {"CO2", "CO3"}
    database.close()


def test_retriever_threshold_and_latest_review_are_refreshed():
    database = DatabaseManager()
    CourseManager(database).add_course(
        "Computer Networks", "CS305", _outcomes()
    )
    question_id = _save_reviewed_example(
        database,
        "CS305",
        "Calculate the shortest path using Dijkstra routing.",
        "CO2",
        "BL3",
    )
    retriever = ApprovedExampleRetriever(
        database, min_similarity=0.8
    )

    assert retriever.retrieve(
        "CS305", "Explain photosynthesis in green plants."
    ) == []
    initial = retriever.retrieve(
        "CS305", "Calculate the shortest path using Dijkstra routing."
    )
    assert initial[0].co_label == "CO2"

    prediction = database.list_question_history(course_code="CS305")[0]
    database.save_approved_labels(
        question_id,
        "CO4",
        "BL4",
        prediction_id=prediction["prediction_id"],
        review_status="corrected",
    )
    refreshed = retriever.retrieve(
        "CS305", "Calculate the shortest path using Dijkstra routing."
    )

    assert refreshed[0].co_label == "CO4"
    assert refreshed[0].bl_label == "BL4"
    database.close()
