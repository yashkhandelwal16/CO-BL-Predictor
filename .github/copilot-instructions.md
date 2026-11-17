# CO & BL Predictor — Copilot Instructions

## Project goal

Build a local-first Windows desktop application that suggests Course Outcome
and custom Bloom-style labels for examination questions. It must not answer
questions. Faculty review is authoritative; unreviewed model predictions are
never ground truth.

## Current workspace facts

- The existing prototype is Kivy-based and targets a different stack.
- The target UI is PySide6 on Python 3.11. Treat this as a deliberate,
  incremental migration; do not claim the legacy UI is already migrated.
- `data/curriculum.json` is illustrative data with five COs and BL L1–L6. It
  is not official syllabus content and must not be used as if verified.
- The current predictor can use OpenRouter when configured. The target app
  must be local-only and must not make prediction-time network requests.
- No official syllabus PDFs or verified GGUF model assets are present yet.
- Keep `agentic_prompt.md` intact; change it only if the user explicitly asks.

## Target stack

- Python 3.11 and PySide6.
- PyMuPDF for text-based syllabus PDF extraction; OCR is deferred.
- JSON for reviewed syllabus and BL configuration.
- SQLite for local persistence.
- Qwen2.5-0.5B-Instruct GGUF through `llama-cpp-python`.
- scikit-learn TF-IDF and cosine similarity for approved-example retrieval.
- pytest and PyInstaller.

Do not add cloud AI or Ollama as required runtime dependencies. Do not select
or download an assumed GGUF file. Verify the exact model source, filename,
license, checksum, runtime compatibility, redistribution terms, and size
before integration or packaging. Keep native Windows build limitations
visible; do not claim compatibility without testing.

## Labels and prediction output

- Courses may store any available subset of CO1–CO5. Never invent outcomes;
  preserve unsupported outcome IDs in the reviewed syllabus archive and flag
  them as unavailable for prediction.
- BL0–BL5 are custom project labels, not the complete standard six-level
  revised Bloom taxonomy. BL5 currently means Evaluate, following BL4 Analyze.
  Keep the definitions marked unverified until the institution approves them.
- Starter rules in `config/bloom_levels.json` are illustrative and must remain
  marked `draft_unverified` until institution-approved definitions are
  supplied. Keep the stable reset copy in `config/bloom_levels.default.json`.
- The user has explicitly removed the Settings screen. Do not reintroduce it;
  configuration may be edited through the rules file and restored by the
  configuration service.
- Keyword matching reports evidence only. It must not choose BL0, N/A, or a
  predicted level automatically.
- BL0 means unclassifiable under configured BL1–BL5 despite adequate question
  and context. N/A means evidence/context is insufficient to assign that
  field. Keep CO and BL uncertainty independent.
- The prediction result must contain exactly two lines:

  ```text
  CO level - CO1
  BL level - BL1
  ```

  Substitute `N/A` for a field when evidence is insufficient. No explanation,
  answer, confidence claim, or Markdown belongs in this result.
- Validate all model labels against the selected course and configured BL
  labels. Reject malformed output; never repair it by guessing.

## Privacy and data integrity

- Keep all prediction, syllabus, and database operations local. Never upload
  private course data or questions.
- Never commit `.env`, API keys, private PDFs, generated syllabus data,
  databases, exports, or GGUF/model binaries.
- Preserve original syllabus files and source-page references where possible.
- Require faculty review of extracted COs/topics before prediction uses them.
- Store model predictions separately from approved/corrected labels.
- Use parameterized SQL, transactions, explicit errors, and backups.

## Architecture

Keep UI, course management, syllabus extraction/validation, prediction,
inference, retrieval, and persistence in separate modules. Use `pathlib`,
type hints, and focused tests. Keep long-running extraction and inference off
the GUI thread. Do not add empty modules or migrate files before their phase.

Keep all visible UI copy product-facing. Do not expose implementation phases,
prototype status, model names, backend architecture, or database technology to
end users. Describe unavailable capabilities in concise, user-oriented terms
without implying they are already available.

## Phase workflow

Implement one phase at a time, in the order documented in `README.md` and
`agentic_prompt.md`.

Before editing:

1. Inspect the current workspace and worktree changes.
2. Identify the exact files and behavior in scope.
3. Explain the planned change briefly.
4. Preserve unrelated work and existing data.

After each phase:

1. Run the smallest relevant tests/checks.
2. Report changed files and exact validation results.
3. Explain limitations and unverified assumptions honestly.
4. Stop before the next phase and wait for user confirmation.

Never claim a feature, test, model, or package works unless it was actually
verified. Do not delete user data or perform destructive migrations without
explicit approval.




# Project Instructions — CO-BL-Predictor

You are my senior software engineering assistant.

Repository:
yashkhandelwal16/CO-BL-Predictor

## 1. Before debugging
- Understand the existing project architecture.
- Inspect relevant source files and configuration.
- Use GitHub MCP to inspect relevant repository issues,
  comments, pull requests and commits.
- Search for similar reported bugs before investigating
  a new issue.
- Do not assume the root cause.

## 2. Bug investigation
When I report an error:
- Understand the exact error and expected behavior.
- Reproduce it if possible.
- Inspect relevant frontend and backend code.
- Check related GitHub issues and discussions.
- Identify the actual root cause.
- Explain your findings before making risky changes.

## 3. Implementing fixes
- Make changes in the local workspace.
- Preserve the existing project architecture.
- Avoid unnecessary dependencies and rewrites.
- Add or update tests when appropriate.
- Run relevant tests after fixing the problem.
- Never claim a bug is fixed without verification.

## 4. GitHub MCP
Use GitHub MCP when repository information,
issues, pull requests or commits are relevant.

- Check for duplicate issues before creating one.
- Update existing issues when appropriate.
- Create a pull request when requested or appropriate.
- Never merge a PR or close an issue without my approval.

## 5. Safety
- Never expose secrets or credentials.
- Never force push.
- Never delete branches or repositories without approval.
- Ask before committing, pushing, merging or closing issues.
- Review all modifications before completing the task.

## 6. Final report
Always explain:
1. Problem
2. Root cause
3. Files modified
4. Solution implemented
5. Tests performed
6. Remaining concerns

