import json
import subprocess
import sys

import pytest

from src.evaluation.evaluate_predictions import load_jsonl
from src.evaluation.metrics import evaluate_predictions


def _examples():
    return [
        {
            "expected_co": "CO1",
            "predicted_co": "CO1",
            "expected_bl": "BL1",
            "predicted_bl": "BL1",
        },
        {
            "expected_co": "CO1",
            "predicted_co": "N/A",
            "expected_bl": "BL1",
            "predicted_bl": "BL2",
        },
        {
            "expected_co": "CO2",
            "predicted_co": "CO2",
            "expected_bl": "BL2",
            "predicted_bl": "N/A",
        },
        {
            "expected_co": "CO2",
            "predicted_co": "CO1",
            "expected_bl": "BL2",
            "predicted_bl": "BL2",
        },
    ]


def test_evaluation_reports_separate_accuracy_class_scores_and_abstention():
    report = evaluate_predictions(_examples())

    assert report["example_count"] == 4
    assert report["co"]["accuracy"] == 0.5
    assert report["co"]["abstention_count"] == 1
    assert report["co"]["abstention_rate"] == 0.25
    assert report["co"]["per_class"]["CO1"] == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "support": 2,
    }
    assert report["bl"]["accuracy"] == 0.5
    assert report["bl"]["abstention_count"] == 1
    assert report["bl"]["abstention_rate"] == 0.25
    assert report["bl"]["per_class"]["BL2"]["support"] == 2
    assert 0.0 <= report["co"]["macro_f1"] <= 1.0
    assert 0.0 <= report["bl"]["macro_f1"] <= 1.0


def test_evaluation_accepts_co5() -> None:
    record = {
        "expected_co": "CO5",
        "predicted_co": "CO5",
        "expected_bl": "BL4",
        "predicted_bl": "BL4",
    }

    report = evaluate_predictions([record])

    assert report["co"]["per_class"]["CO5"]["support"] == 1
    assert report["co"]["accuracy"] == 1.0


@pytest.mark.parametrize(
    ("records", "message"),
    [
        ([], "At least one labelled"),
        ([{"expected_co": "CO1"}], "missing"),
        (
            [
                {
                    **_examples()[0],
                    "expected_co": "N/A",
                }
            ],
            "invalid expected CO",
        ),
        (
            [
                {
                    **_examples()[0],
                    "predicted_co": "BL1",
                }
            ],
            "invalid predicted CO",
        ),
    ],
)
def test_evaluation_rejects_empty_or_invalid_labelled_data(records, message):
    with pytest.raises(ValueError, match=message):
        evaluate_predictions(records)


def test_jsonl_loader_reads_utf8_examples_and_reports_malformed_lines(tmp_path):
    source = tmp_path / "faculty-labelled.jsonl"
    records = _examples()
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert load_jsonl(source) == records

    source.write_text('{"expected_co":\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_jsonl(source)


def test_jsonl_loader_rejects_non_object_rows(tmp_path):
    source = tmp_path / "invalid.jsonl"
    source.write_text('["CO1", "BL1"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_jsonl(source)


def test_evaluation_command_writes_a_json_metrics_report(tmp_path):
    source = tmp_path / "held-out.jsonl"
    output = tmp_path / "metrics.json"
    source.write_text(
        json.dumps(_examples()[0]) + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_predictions",
            str(source),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["example_count"] == 1
    assert report["co"]["accuracy"] == 1.0
    assert report["bl"]["accuracy"] == 1.0
