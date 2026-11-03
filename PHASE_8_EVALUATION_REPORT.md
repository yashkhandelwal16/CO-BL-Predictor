# Phase 8 — Accuracy Evaluation Report

**Status: BLOCKED — faculty-verified held-out data is not available.**

## Dataset and methodology

No faculty-verified evaluation dataset was present in the workspace. The
illustrative `data/curriculum.json` is not verified syllabus data and was not
used. The application's local SQLite database contains 21 questions and zero
approved-label records, so none of those questions can serve as verified
ground truth.

Consequently, no held-out split could be defined, leakage into approved-example
retrieval could not be ruled out for any candidate dataset, and no predictions
were generated for this evaluation. No model or decision thresholds were tuned
on held-out examples.

The requested dataset should include course name/code, official CO
descriptions, reviewed syllabus topics, academic question, faculty-approved
CO and BL labels, and question validity. Keep it local and provide a held-out
set that is not available to retrieval or tuning. Labels and course content
should be supplied only through a local file or local workspace path.

## Configuration results

All requested configurations are **not measured**. Accuracy, precision,
recall, macro F1, confusion matrices, accepted-prediction accuracy, latency,
and per-configuration failure counts are therefore **N/A**, not zero.

| Configuration | Description | CO metrics | BL metrics | Confusion matrices | Latency |
|---|---|---|---|---|---|
| A | Existing Qwen-only prediction | N/A | N/A | N/A | N/A |
| B | Qwen + syllabus-aware NLP | N/A | N/A | N/A | N/A |
| C | Qwen + NLP + faculty-approved examples | N/A | N/A | N/A | N/A |
| D | Full hybrid pipeline with DDGS research | N/A | N/A | N/A | N/A |

No accuracy improvement is claimed.

## Additional requested metrics

| Metric | Result | Reason |
|---|---|---|
| Invalid-question rejection rate | N/A | No labelled held-out questions |
| CO and BL abstention rates | N/A | No configuration predictions |
| Accuracy among accepted predictions | N/A | No accepted held-out predictions |
| DDGS source relevance | N/A | No evaluation queries or relevance labels |
| Web extraction success rate | N/A | No evaluation pages fetched |
| Prediction latency | N/A | No evaluation runs were performed |
| Offline fallback success rate | N/A as a measured rate | The mocked worker regression passes, but it is not a representative benchmark |

## Test and compatibility evidence

The full regression suite passed: **228 tests**. Existing unit tests exercise
mocked prediction, research, extraction, error/fallback, and UI behaviors; they
are software regression tests, not faculty-labelled accuracy measurements.
Live DDGS behavior and real-model performance were not benchmarked in this
evaluation.

| Requested scenario | Regression evidence | Held-out/real-world result |
|---|---|---|
| Normal academic questions and single CO evidence | Predictor and hybrid decision unit tests | Not measured |
| Multiple plausible COs and disagreement | Query-generator and hybrid abstention tests | Not measured |
| Irrelevant questions and invalid-input rejection | Predictor and hybrid invalid-question tests | Not measured on a faculty-labelled set |
| Random names/meaningless text | Some input guards are tested; no dedicated representative noise set | Not measured |
| Ambiguous cognitive demand | Independent CO/BL uncertainty tests | Not measured |
| BL1–BL6 | Legacy labels and a hybrid BL6 path are unit-tested; no held-out class-wise results | Not measured |
| Different course sizes | Multiple candidate CO query generation is tested; no scale benchmark | Not measured |
| Missing syllabus/context | Predictor tests reject missing required context | Not measured on held-out data |
| Internet unavailable, DDGS errors/blocking | Mocked search failure and safe-error tests | Live provider behavior not measured |
| Web extraction failures | Mocked timeout, blocked, empty, unsafe redirect, and snippet fallback tests | Live extraction success rate not measured |
| Qwen server/inference failure | Mocked inference-error handling is covered | Real server reliability/latency not measured |
| Application restart | Database persistence and local toggle persistence tests | Accuracy after restart not applicable without data |
| Existing SQLite compatibility | v4-to-v5 migration verified on a temporary copy of the actual local database | Passed; original database unchanged |

The locally installed database was schema v4. A temporary copy was migrated
through the current v5 migration: integrity check passed, legacy course,
question, prediction, and approved-label row counts were preserved, and the
hybrid-predictions table was created. The original database was not modified.

## Failure analysis and limitations

The primary blocker is the absence of ground truth: zero approved labels are
available for a held-out test set. This prevents meaningful CO/BL accuracy
claims, confusion matrices, error analysis, and comparisons between A–D.
Mocked service tests cannot establish DDGS relevance, page extraction rates,
or real offline fallback reliability under representative conditions.

The existing evaluation utility currently reports paired-label classification
metrics, but does not execute the four pipeline variants or produce the full
requested operational metrics and confusion matrices. Those outputs should be
added alongside the evaluation runner once the dataset and split policy are
available. It currently accepts only custom BL0–BL5, so it cannot evaluate
revised BL6. Revised BL1–BL6 must remain distinct from the legacy custom
BL0–BL5 taxonomy; an institution-approved mapping or separate per-taxonomy
reporting is required before comparing Bloom results across configurations.

## Recommendations and next step

1. Supply a faculty-verified dataset and identify the held-out portion.
2. Confirm which validity labels are intended and how invalid questions should
   be scored (rejection versus classification).
3. Keep held-out examples out of approved-example retrieval and all tuning.
4. Run all four configurations on exactly the same examples and record
   per-example outputs, abstentions, latency, DDGS source relevance, extraction
   status, and fallback outcomes.
5. Report CO and BL metrics independently, with confusion matrices, class
   supports, and failure analysis; treat model confidence as uncalibrated.

Until that dataset is available, Phase 8 accuracy evaluation remains
incomplete.
