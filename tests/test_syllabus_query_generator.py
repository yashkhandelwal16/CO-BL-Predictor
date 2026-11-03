import pytest

from src.research import SyllabusAwareQueryGenerator


def test_java_question_ranks_official_co_and_builds_contextual_query():
    outcomes = {
        "CO1": "Recall Java syntax and primitive data types.",
        "CO2": "Implement inheritance, polymorphism, and encapsulation in Java.",
        "CO3": "Analyze exception handling and file operations.",
    }
    original_description = outcomes["CO2"]
    result = SyllabusAwareQueryGenerator().generate(
        "Implement a Java program using inheritance.",
        "Object Oriented Programming using Java",
        "CS201",
        outcomes,
        {
            "Unit II: Object Oriented Programming": [
                "Inheritance and polymorphism",
                "Method overriding",
            ],
            "Unit III: Robust Programs": ["Exception handling"],
        },
        ["Java", "inheritance", "polymorphism"],
        "Apply",
    )

    assert result.plausible_co_candidates[0] == "CO2"
    assert result.co_rankings[0].description == original_description
    assert result.extracted_concepts
    assert "inheritance" in result.technical_terminology
    assert [topic.topic for topic in result.relevant_syllabus_topics] == [
        "Inheritance and polymorphism",
    ]
    assert result.subject_area == "Object Oriented Programming using Java"
    assert result.cognitive_action == "Apply"
    assert result.actual_cognitive_task.startswith("Implement")
    query = result.generated_queries[0]
    assert query.co_id == "CO2"
    assert "Implement a Java program using inheritance." in query.query
    assert original_description in query.query
    assert "Object Oriented Programming using Java" in query.query
    assert "Inheritance and polymorphism" in query.query
    assert "academic learning objectives" in query.query
    assert "web results do not assign or confirm the CO" in query.reason


@pytest.mark.parametrize(
    (
        "question",
        "course_name",
        "course_code",
        "outcomes",
        "topics",
        "concepts",
        "action",
        "expected_co",
    ),
    [
        (
            "How does a router forward packets using a routing table?",
            "Computer Networks",
            "CS305",
            {
                "CO1": "Explain network models and protocol layering.",
                "CO2": "Analyze routing algorithms and packet forwarding.",
                "CO3": "Configure transport protocols and congestion control.",
            },
            {
                "Unit I: Network Fundamentals": ["Network models", "Protocol layering"],
                "Unit III: Network Layer": [
                    "Routing algorithms",
                    "Packet forwarding and routing tables",
                ],
            },
            ["router", "routing table", "packet forwarding"],
            "Understand",
            "CO2",
        ),
        (
            "Compare photosynthesis and cellular respiration in plants.",
            "General Biology",
            "BIO110",
            {
                "CO1": "Describe cell structures and organelles.",
                "CO2": "Explain photosynthesis and chloroplast function.",
                "CO3": "Compare cellular respiration and energy production.",
            },
            {
                "Unit II: Plant Biology": ["Photosynthesis", "Chloroplasts"],
                "Unit III: Cell Metabolism": [
                    "Cellular respiration",
                    "Energy production",
                ],
            },
            ["photosynthesis", "cellular respiration"],
            "Analyze",
            "CO3",
        ),
    ],
)
def test_subject_and_question_format_changes_preserve_relevant_ranking(
    question,
    course_name,
    course_code,
    outcomes,
    topics,
    concepts,
    action,
    expected_co,
):
    result = SyllabusAwareQueryGenerator().generate(
        question,
        course_name,
        course_code,
        outcomes,
        topics,
        concepts,
        action,
    )

    assert result.plausible_co_candidates[0] == expected_co
    assert result.generated_queries[0].co_id == expected_co
    assert result.cognitive_action == action
    assert result.relevant_syllabus_topics


def test_multiple_plausible_cos_are_ranked_and_each_gets_its_own_query():
    outcomes = {
        "CO1": "Apply graph traversal algorithms such as BFS and DFS.",
        "CO2": "Analyze graph traversal complexity and shortest paths.",
        "CO3": "Design relational databases and SQL queries.",
    }
    result = SyllabusAwareQueryGenerator().generate(
        "Analyze the time complexity of BFS and DFS graph traversal algorithms.",
        "Data Structures and Algorithms",
        "CS301",
        outcomes,
        {
            "Unit IV: Graphs": [
                "Breadth-first search BFS",
                "Depth-first search DFS",
                "Shortest path algorithms",
            ]
        },
        ["BFS", "DFS", "graph traversal"],
        "Analyze",
    )

    assert set(result.plausible_co_candidates) == {"CO1", "CO2"}
    assert {query.co_id for query in result.generated_queries} == set(
        result.plausible_co_candidates
    )
    assert all(query.query for query in result.generated_queries)
    assert all(query.reason for query in result.generated_queries)
    assert "relational databases" not in result.generated_queries[0].query


def test_unrelated_question_returns_rankings_but_no_plausible_cos_or_queries():
    outcomes = {
        "CO1": "Explain relational database models and ER diagrams.",
        "CO2": "Apply SQL queries and database normalization.",
    }
    result = SyllabusAwareQueryGenerator().generate(
        "Describe the migration patterns of monarch butterflies.",
        "Database Management Systems",
        "CS402",
        outcomes,
        {"Unit I: Relational Models": ["ER diagrams", "Relational schemas"]},
        [],
        "Understand",
    )

    assert {ranking.co_id for ranking in result.co_rankings} == set(outcomes)
    assert result.plausible_co_candidates == ()
    assert result.generated_queries == ()


def test_large_course_keeps_all_official_outcomes_and_caps_candidate_queries():
    outcomes = {
        f"CO{index}": "Apply graph traversal algorithms using BFS and DFS."
        for index in range(1, 16)
    }
    result = SyllabusAwareQueryGenerator(max_candidates=2).generate(
        "Apply graph traversal algorithms using BFS and DFS.",
        "Data Structures and Algorithms",
        "CS301",
        outcomes,
        {"Unit IV: Graphs": ["Graph traversal algorithms", "BFS and DFS"]},
        ["graph traversal", "BFS", "DFS"],
        "Apply",
    )

    assert len(result.co_rankings) == len(outcomes)
    assert {ranking.co_id for ranking in result.co_rankings} == set(outcomes)
    assert result.plausible_co_candidates == ("CO1", "CO2")
    assert len(result.generated_queries) == 2
    assert set(result.plausible_co_candidates) <= set(outcomes)


def test_extracted_syllabus_unit_records_are_supported():
    result = SyllabusAwareQueryGenerator().generate(
        "Explain the ACID properties in database transactions.",
        "Database Management Systems",
        "CS402",
        {"CO4": "Analyze transactions, ACID properties, and concurrency control."},
        [
            {
                "unit_number": 4,
                "unit_title": "Transaction Processing",
                "topics": ["ACID properties", "Serializability"],
            }
        ],
        ["transactions", "ACID properties"],
        "Understand",
    )

    assert result.plausible_co_candidates == ("CO4",)
    assert result.relevant_syllabus_topics[0].unit == "Transaction Processing"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("academic_question", " ", "Academic question"),
        ("course_name", "", "Course name"),
        ("course_code", "", "Course code"),
        ("detected_cognitive_action", "", "Detected cognitive action"),
    ],
)
def test_required_inputs_are_validated(field, value, message):
    inputs = {
        "academic_question": "Explain inheritance.",
        "course_name": "Java Programming",
        "course_code": "CS201",
        "course_outcomes": {"CO1": "Explain inheritance in Java."},
        "unit_topics": {},
        "relevant_technical_concepts": [],
        "detected_cognitive_action": "Understand",
    }
    inputs[field] = value

    with pytest.raises(ValueError, match=message):
        SyllabusAwareQueryGenerator().generate(**inputs)


def test_outcome_descriptions_are_required_and_unmodified():
    original = "  Apply database normalization without changing this text.  "
    result = SyllabusAwareQueryGenerator().generate(
        "Apply database normalization.",
        "Database Systems",
        "DB101",
        {"CO7": original},
        {},
        ["database normalization"],
        "Apply",
    )

    assert result.co_rankings[0].description == original
    assert result.plausible_co_candidates == ("CO7",)

    with pytest.raises(ValueError, match="description"):
        SyllabusAwareQueryGenerator().generate(
            "Question",
            "Course",
            "C1",
            {"CO1": "  "},
            {},
            [],
            "Understand",
        )


def test_generator_options_are_validated():
    with pytest.raises(ValueError, match="candidate count"):
        SyllabusAwareQueryGenerator(max_candidates=0)
    with pytest.raises(ValueError, match="threshold"):
        SyllabusAwareQueryGenerator(min_co_similarity=1.1)
