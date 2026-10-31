from collections.abc import Iterable, Mapping
from typing import Any

from sklearn.metrics import accuracy_score, precision_recall_fscore_support


CO_LABELS = ("CO1", "CO2", "CO3", "CO4", "CO5")
BL_LABELS = ("BL0", "BL1", "BL2", "BL3", "BL4", "BL5")


def _measure_field(
    examples: list[Mapping[str, Any]],
    *,
    expected_key: str,
    predicted_key: str,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    expected = [str(example[expected_key]) for example in examples]
    predicted = [str(example[predicted_key]) for example in examples]

    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=list(labels),
        zero_division=0,
    )
    abstentions = sum(label == "N/A" for label in predicted)
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "abstention_count": abstentions,
        "abstention_rate": abstentions / len(examples),
        "macro_f1": float(f1.mean()),
        "per_class": {
            label: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }


def evaluate_predictions(
    examples: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure CO/BL accuracy, per-class scores, macro-F1 and abstentions."""
    records = list(examples)
    if not records:
        raise ValueError("At least one labelled evaluation example is required.")

    required_keys = {
        "expected_co",
        "predicted_co",
        "expected_bl",
        "predicted_bl",
    }
    for index, example in enumerate(records, start=1):
        missing = required_keys - example.keys()
        if missing:
            raise ValueError(
                f"Evaluation example {index} is missing: "
                f"{', '.join(sorted(missing))}."
            )
        for key in required_keys:
            if not isinstance(example[key], str):
                raise ValueError(
                    f"Evaluation example {index} field {key} must be text."
                )
        if example["expected_co"] not in CO_LABELS:
            raise ValueError(
                f"Evaluation example {index} has an invalid expected CO label."
            )
        if example["expected_bl"] not in BL_LABELS:
            raise ValueError(
                f"Evaluation example {index} has an invalid expected BL label."
            )
        if example["predicted_co"] not in (*CO_LABELS, "N/A"):
            raise ValueError(
                f"Evaluation example {index} has an invalid predicted CO label."
            )
        if example["predicted_bl"] not in (*BL_LABELS, "N/A"):
            raise ValueError(
                f"Evaluation example {index} has an invalid predicted BL label."
            )

    return {
        "example_count": len(records),
        "co": _measure_field(
            records,
            expected_key="expected_co",
            predicted_key="predicted_co",
            labels=CO_LABELS,
        ),
        "bl": _measure_field(
            records,
            expected_key="expected_bl",
            predicted_key="predicted_bl",
            labels=BL_LABELS,
        ),
    }
