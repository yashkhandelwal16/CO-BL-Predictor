from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BloomLevel:
    id: str
    name: str
    definition: str
    cognitive_processes: tuple[str, ...]
    examples: tuple[str, ...]


BLOOM_LEVELS = (
    BloomLevel(
        "BL1",
        "Remember",
        "Recall previously learned facts, definitions, terminology, formulas and information.",
        (
            "remember",
            "recognize",
            "recall",
            "identify",
            "list",
            "name",
            "define",
            "reproduce",
        ),
        (
            "Define inheritance.",
            "List features of Java.",
            "What is a constructor?",
        ),
    ),
    BloomLevel(
        "BL2",
        "Understand",
        "Demonstrate understanding of concepts by explaining, interpreting, summarizing, classifying or comparing their meaning.",
        (
            "explain",
            "describe",
            "summarize",
            "interpret",
            "classify",
            "illustrate",
            "compare",
        ),
        (
            "Explain polymorphism.",
            "Differentiate between JDK and JRE.",
            "Describe the working of inheritance.",
            "Illustrate the relationship between a class and an object.",
        ),
    ),
    BloomLevel(
        "BL3",
        "Apply",
        "Use previously learned knowledge, concepts, methods, rules or algorithms to solve a problem or perform a task.",
        (
            "apply",
            "implement",
            "execute",
            "calculate",
            "demonstrate",
            "solve",
            "use",
        ),
        (
            "Write a Java program to implement inheritance.",
            "Implement a stack using an array.",
            "Apply a sorting algorithm to a given array.",
        ),
    ),
    BloomLevel(
        "BL4",
        "Analyze",
        "Break information, concepts or systems into components and examine their relationships, patterns, structure and logical connections.",
        (
            "analyze",
            "examine",
            "differentiate",
            "organize",
            "investigate",
            "identify relationships",
        ),
        (
            "Analyze the relationship between two classes.",
            "Examine the time complexity of an algorithm.",
            "Identify logical errors in a program.",
        ),
    ),
    BloomLevel(
        "BL5",
        "Evaluate",
        "Make judgments or decisions based on criteria, standards, evidence and logical reasoning.",
        (
            "evaluate",
            "justify",
            "validate",
            "critique",
            "assess",
            "defend",
            "judge",
            "recommend",
        ),
        (
            "Evaluate two sorting algorithms based on performance.",
            "Justify a database design.",
            "Assess whether an algorithm meets the given requirements.",
        ),
    ),
    BloomLevel(
        "BL6",
        "Create",
        "Combine existing knowledge and concepts to design, construct, formulate or develop a new solution, model, product or approach.",
        (
            "design",
            "develop",
            "construct",
            "formulate",
            "propose",
            "create",
            "plan",
        ),
        (
            "Design a database for a hospital management system.",
            "Develop a new algorithm for a given problem.",
            "Propose a new system architecture.",
        ),
    ),
)

BLOOM_CLASSIFICATION_RULES = (
    "Never determine the Bloom level using only an action verb; classify the complete question and expected student response.",
    "Question difficulty and question length do not determine the Bloom level.",
    "A programming question is not automatically BL6.",
    "BL2 is understanding a concept; BL3 is using knowledge to perform a task.",
    "BL3 applies a known method; BL4 examines structures, relationships or patterns.",
    "BL4 analyzes evidence; BL5 judges using criteria, standards or evidence.",
    "BL5 evaluates an existing solution; BL6 creates a new solution.",
    "The words 'explain' and 'design' do not determine a level on their own.",
    "'Compare' may be BL2 or BL4 depending on the expected depth of analysis.",
    "When multiple cognitive operations are present, identify the primary expected learning objective.",
    "Do not automatically select the highest cognitive level.",
    "Return N/A when evidence about the expected cognitive work is insufficient.",
)

BLOOM_LEVEL_IDS = tuple(level.id for level in BLOOM_LEVELS)
BLOOM_LABELS = frozenset(BLOOM_LEVEL_IDS)

_WORD_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
_LEADING_INSTRUCTION = re.compile(
    r"^\s*(?:please\s+)?(?P<instruction>[a-z]+)\b", re.IGNORECASE
)
_REMEMBER_CUES = frozenset(
    {
        "define",
        "identify",
        "list",
        "name",
        "recall",
        "recognize",
        "reproduce",
        "state",
    }
)
_UNDERSTAND_CUES = frozenset(
    {
        "classify",
        "describe",
        "differentiate",
        "explain",
        "illustrate",
        "interpret",
        "summarize",
        "summarise",
    }
)
_APPLY_CUES = frozenset(
    {
        "apply",
        "calculate",
        "compute",
        "demonstrate",
        "execute",
        "implement",
        "solve",
        "use",
        "write",
    }
)
_ANALYZE_CUES = frozenset(
    {"analyze", "analyse", "examine", "investigate", "organize", "organise"}
)
_EVALUATE_CUES = frozenset(
    {
        "assess",
        "critique",
        "defend",
        "evaluate",
        "judge",
        "justify",
        "recommend",
        "validate",
    }
)
_CREATE_CUES = frozenset(
    {"construct", "create", "design", "develop", "formulate", "plan", "propose"}
)
_CRITERIA_CUES = frozenset(
    {
        "accuracy",
        "criteria",
        "criterion",
        "correctness",
        "effectiveness",
        "efficiency",
        "performance",
        "quality",
        "requirements",
        "standard",
        "standards",
    }
)
_COMPUTATION_TARGET_CUES = frozenset(
    {"arithmetic", "equation", "expression", "numeric", "value"}
)
_ANALYSIS_CUES = frozenset(
    {
        "complexity",
        "error",
        "errors",
        "interaction",
        "interactions",
        "pattern",
        "patterns",
        "relationship",
        "relationships",
        "structure",
        "structures",
    }
)
_CREATE_OBJECT_CUES = frozenset(
    {
        "algorithm",
        "approach",
        "architecture",
        "database",
        "framework",
        "model",
        "plan",
        "product",
        "program",
        "procedure",
        "solution",
        "system",
        "workflow",
    }
)


def format_bloom_knowledge() -> str:
    sections = []
    for level in BLOOM_LEVELS:
        processes = ", ".join(level.cognitive_processes)
        examples = "\n".join(f"- {example}" for example in level.examples)
        sections.append(
            f"{level.id} – {level.name.upper()}\n"
            f"Definition: {level.definition}\n"
            f"Cognitive processes: {processes}\n"
            f"Examples:\n{examples}"
        )
    rules = "\n".join(
        f"{number}. {rule}"
        for number, rule in enumerate(BLOOM_CLASSIFICATION_RULES, start=1)
    )
    return (
        "Revised Bloom's Taxonomy:\n"
        + "\n\n".join(sections)
        + "\n\nClassification rules:\n"
        + rules
    )


def detect_cognitive_action(question: str) -> str:
    words = {word.casefold() for word in _WORD_PATTERN.findall(question)}
    for level in BLOOM_LEVELS:
        for process in level.cognitive_processes:
            process_words = process.casefold().split()
            if all(word in words for word in process_words):
                return process
    if "what" in words and ("is" in words or "are" in words):
        return "recall"
    return "not specified"


def classify_bloom_level(question: str) -> str:
    """Classify the primary requested work without treating verbs as labels."""
    normalized = " ".join(question.split()).casefold()
    if not normalized:
        return "N/A"

    words = {word.casefold() for word in _WORD_PATTERN.findall(normalized)}
    instruction_match = _LEADING_INSTRUCTION.match(normalized)
    instruction = (
        instruction_match.group("instruction").casefold()
        if instruction_match is not None
        else ""
    )
    if instruction == "evaluate" and words.intersection(
        _COMPUTATION_TARGET_CUES
    ):
        return "BL3"
    if instruction == "evaluate":
        return (
            "BL5"
            if words.intersection(_CRITERIA_CUES | {"evidence", "tradeoff"})
            else "N/A"
        )
    if instruction in _EVALUATE_CUES:
        return "BL5"
    if (
        instruction in {"compare", "differentiate"}
        and words.intersection(_CRITERIA_CUES)
    ):
        return "BL4"
    if instruction in _ANALYZE_CUES:
        return "BL4"
    if (
        instruction == "identify"
        and words.intersection(_ANALYSIS_CUES)
    ):
        return "BL4"
    if (
        instruction == "explain"
        and words.intersection(_CRITERIA_CUES)
        and words.intersection(
            _CREATE_OBJECT_CUES | _ANALYSIS_CUES | {"design", "algorithm"}
        )
    ):
        return "BL5"
    if instruction == "explain" and words.intersection(_ANALYSIS_CUES):
        return "BL4"
    if instruction in _UNDERSTAND_CUES:
        return "BL2"
    if instruction == "compare":
        return "BL4" if words.intersection(_ANALYSIS_CUES) else "BL2"
    if instruction in _APPLY_CUES:
        return "BL3"
    if instruction in _CREATE_CUES:
        if words.intersection(_CREATE_OBJECT_CUES):
            return "BL6"
        return "N/A"
    if instruction in _REMEMBER_CUES:
        if (
            instruction == "identify"
            and words.intersection(_ANALYSIS_CUES)
        ):
            return "BL4"
        return "BL1"
    if normalized.startswith(("what is ", "what are ", "who is ")):
        return "BL1"
    return "N/A"
