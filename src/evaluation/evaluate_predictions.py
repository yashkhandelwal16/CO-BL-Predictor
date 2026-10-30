import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import evaluate_predictions


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    examples: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {error.msg}."
                    ) from error
                if not isinstance(example, dict):
                    raise ValueError(
                        f"Evaluation line {line_number} must contain a JSON object."
                    )
                examples.append(example)
    except OSError as error:
        raise ValueError(f"Could not read evaluation file {source}: {error}") from error
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate CO/BL labels against faculty-reviewed held-out data."
    )
    parser.add_argument("input", type=Path, help="UTF-8 JSON Lines evaluation file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON metrics report (defaults to stdout)",
    )
    arguments = parser.parse_args()

    try:
        report = evaluate_predictions(load_jsonl(arguments.input))
        rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        if arguments.output is None:
            print(rendered, end="")
        else:
            arguments.output.write_text(rendered, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
