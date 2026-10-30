# CO & BL Predictor

A local-first Windows desktop application for suggesting Course Outcome (CO)
and Bloom-style (BL) labels for examination questions. The application is an
academic aid: faculty review remains authoritative, and the model must never
answer the examination question.

## Initial project audit (Phase 0)

This records the workspace state at the start of the project, before the
migration began:

- `main.py` contains a Kivy desktop UI.
- `predictor.py` contains keyword prediction and an optional OpenRouter client.
  If `OPENROUTER_API_KEY` is configured, the current prototype can send a
  question to that online service. Do not enter confidential questions into
  this version.
- `data/curriculum.json` contains example subjects with five COs and BL levels
  L1–L6. These entries are not verified official syllabus data.
- The existing virtual environment uses Python 3.12.4. The Python launcher
  currently lists no Python 3.11 installation.
- There are no official syllabus PDFs, model files, database, or automated
  tests in the workspace.

These are historical audit findings, not a description of the current
application.

## Target technology and requirements

- Python 3.11 and PySide6 for the Windows desktop UI.
- PyMuPDF for text-based syllabus PDF extraction; scanned-PDF OCR is deferred.
- JSON for reviewed syllabus and BL configuration.
- SQLite for local course, question, prediction, and review data.
- Qwen2.5-0.5B-Instruct in GGUF format via `llama-cpp-python` for local
  inference.
- scikit-learn for TF-IDF and cosine-similarity retrieval.
- pytest for automated tests and PyInstaller for Windows packaging.

The app must not require a cloud API or Ollama. A model candidate is specified,
but its exact repository, GGUF filename, license, checksum, runtime
compatibility, and redistribution terms have not yet been verified. Do not
download or bundle an assumed model file; resolve and document those details
before model integration and redistribution.

`requirements.txt` lists the target Python packages. In particular,
`llama-cpp-python` includes native components whose compatible Windows build
must be verified during the model-integration phase. These dependencies have
not yet been installed or certified in this workspace.

## Architecture and data flow

Keep the UI, syllabus processing, prediction, persistence, and model inference
in separate modules:

1. The PySide6 UI collects a question and selected course.
2. The syllabus service extracts PDF text and produces structured JSON for
   faculty review; it must preserve the source PDF and flag uncertain fields.
3. Prediction receives only the selected course's reviewed CO definitions,
   relevant reviewed syllabus topics, configured BL rules, and approved
   same-course examples.
4. The local inference service returns structured candidate labels. A
   validator rejects labels outside the selected course's CO1–CO4 and BL0–BL4
   sets; it does not repair malformed output by guessing.
5. SQLite stores course/syllabus versions, questions, predictions, and
   faculty-approved labels separately. The UI displays only the required
   prediction labels and review state.

Suggested structure, to create as each phase needs it (not all at once):

```text
main.py
requirements.txt
README.md
.github/
  copilot-instructions.md
config/
data/
  syllabi/       # local input; not committed
  parsed_json/   # generated/local data; not committed
  exports/       # generated/local data; not committed
models/          # local GGUF assets; not committed
src/
  ui/
  courses/
  syllabus/
  prediction/
  ai/
  nlp/
  database/
  utils/
tests/
scripts/
packaging/
```

The existing files should be migrated incrementally. Do not create empty
modules or move the prototype wholesale before the relevant phase.

## Label and output policy

- Each course uses exactly CO1, CO2, CO3, and CO4. If the official syllabus
  does not support that count, flag the course for faculty review; never
  truncate or invent official outcomes.
- The project uses custom BL0–BL4 labels, not the standard six-level revised
  Bloom taxonomy. Faculty/institution definitions must be documented before
  the rules are treated as official.
- BL0 means the question is unclassifiable under the configured BL1–BL4
  scheme despite adequate question/context. N/A means evidence or context is
  insufficient to assign that field. CO and BL uncertainty are independent.
- The prediction output contains exactly these two lines:

  ```text
  CO level-CO1
  BL level - BL1
  ```

  Use `N/A` in either field when its evidence is insufficient. Do not include
  explanations, generated answers, or confidence claims in the prediction
  result.
- Predictions are suggestions. Only faculty-approved labels are ground truth
  for retrieval or evaluation.

## Privacy and Git

The application is designed to work locally and must not upload syllabi,
questions, or other user data. Keep private PDFs, generated parsed data,
database files, exports, and model binaries out of Git.

## Development phases

0. Workspace audit, requirements, project rules, and architecture — complete.
1. Python 3.11 environment and PySide6 GUI.
2. Course management and CO entry.
3. PDF extraction and reviewable JSON.
4. Configurable custom BL0–BL4 rules.
5. SQLite persistence and faculty review.
6. Local Qwen/llama.cpp compatibility and model integration.
7. Validated CO/BL prediction and N/A handling.
8. TF-IDF retrieval of approved same-course examples.
9. Automated tests and honest evaluation on faculty-labelled data.
10. Windows packaging and offline clean-machine testing.

Android is a later, separate feasibility effort after the Windows application
is stable. Do not assume that the desktop model runtime can run in an Android
package.

## Run the application

The desktop application uses PySide6. The current machine has Python
3.12.4, and the UI has been tested in the existing `venv` with that interpreter.
Python 3.11 is the target version but is not currently installed; this means
the target interpreter has not yet been verified.

Run the app from PowerShell in the project folder:

```powershell
.\venv\Scripts\python.exe main.py
```

Run the project tests:

```powershell
.\venv\Scripts\python.exe -m pytest
```

The Courses screen supports create, edit, delete, and name/code search. It
requires a name, a unique course code, and four separate CO descriptions.
Courses and their CO descriptions persist between application launches.
Selecting a syllabus PDF in the course form extracts clearly labelled COs for
faculty to review and edit before saving.

The Syllabus screen extracts text from text-based PDFs and opens an editable
review of course details, units, topics, and outcomes. Reviewed JSON and
versioned syllabus metadata are saved in the per-user application data folder;
the original PDF remains in its original location. Unclear or missing fields
are flagged. Scanned PDFs without selectable text are reported; OCR is not
included. The History screen displays saved question records and offers data
backup and JSON export. Prediction and faculty review workflows are not yet
available in the interface.

On Windows, the application data folder is
`%LOCALAPPDATA%\\CO-BL-Predictor`. It contains the database and reviewed
syllabus JSON files. Use **Back up data** from History to create a database
backup, or **Export data** to save a JSON copy of the stored records. These
files may contain private academic information; store and share them
appropriately.

PyMuPDF is distributed under AGPL-3.0 or a commercial license. Review its
licensing terms before distributing this application.

The custom BL0–BL4 keyword definitions are in `config/bloom_levels.json`.
They are starter examples, marked as unverified, not official institutional
rules. Update keywords there and restart the app; the configuration is
validated on startup. BL keywords provide evidence only and do not determine a
final classification by themselves. There is intentionally no Settings screen.

To create the intended Python 3.11 environment after Python 3.11 is installed:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The PySide6 UI dependencies were installed into the existing ignored `venv`.
The full target dependency set and Python 3.11 compatibility have not yet been
verified. Do not install the native inference dependency or download a model
until the exact model/runtime combination has been checked in Phase 6.

## Phase status

- Phase 0 — Workspace audit, requirements, project rules, and architecture:
  complete.
- Phase 1 — PySide6 GUI: implemented and smoke-tested under Python 3.12.4;
  Python 3.11 setup remains blocked until that interpreter is installed.
- Phase 2 — Course management: create/edit/delete/search and independent
  CO1–CO4 fields implemented and tested.
- Phase 3 — PDF extraction and reviewable JSON: implemented for text-based
  PDFs, with editable details, unit/topic and outcome tables, page references,
  warnings, validation, and source-file preservation.
- Phase 4 — Custom BL0–BL4 configuration and keyword evidence: implemented.
  Definitions remain draft/unverified until institution-approved rules are
  supplied. Settings UI remains excluded.
- Phase 5 — SQLite persistence: versioned schema, course and syllabus
  persistence, question/prediction/review storage, History display, backup,
  export, and restart behavior implemented and tested.
- Phases 6–10 — Not started.
