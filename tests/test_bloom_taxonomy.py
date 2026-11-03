import pytest

from src.prediction.bloom_taxonomy import (
    BLOOM_LEVELS,
    classify_bloom_level,
    format_bloom_knowledge,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Define inheritance.", "BL1"),
        ("What is a constructor?", "BL1"),
        ("Explain polymorphism.", "BL2"),
        ("Differentiate between JDK and JRE.", "BL2"),
        (
            "Write a Java program to implement inheritance.",
            "BL3",
        ),
        ("Implement a stack using an array.", "BL3"),
        (
            "Analyze the relationship between two classes.",
            "BL4",
        ),
        ("Examine the time complexity of an algorithm.", "BL4"),
        (
            "Compare two sorting algorithms based on performance.",
            "BL4",
        ),
        (
            "Evaluate two sorting algorithms based on performance.",
            "BL5",
        ),
        ("Evaluate the following expression: 2 + 3.", "BL3"),
        ("Evaluate an algorithm.", "N/A"),
        (
            "Explain why this design violates the stated performance requirements.",
            "BL5",
        ),
        (
            "Design a database for a hospital management system.",
            "BL6",
        ),
        (
            "Explain how to design a database for a hospital.",
            "BL2",
        ),
        ("A long, difficult programming question.", "N/A"),
        ("", "N/A"),
    ],
)
def test_classifies_full_task_not_just_action_verb(question, expected):
    assert classify_bloom_level(question) == expected


def test_knowledge_base_contains_all_six_levels_and_classification_rules():
    knowledge = format_bloom_knowledge()

    assert [level.id for level in BLOOM_LEVELS] == [
        "BL1",
        "BL2",
        "BL3",
        "BL4",
        "BL5",
        "BL6",
    ]
    assert "BL6 – CREATE" in knowledge
    assert "Never determine the Bloom level using only an action verb" in knowledge
    assert "Question difficulty and question length do not determine" in knowledge
    assert "Do not automatically select the highest cognitive level" in knowledge
