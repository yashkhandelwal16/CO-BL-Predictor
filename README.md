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

The app must not require a cloud API or Ollama. Phase 6 identifies the official
GGUF file and its Apache-2.0 license and implements a local inference client.
Model weights are not bundled in the source repository; download them
separately as described below.

`requirements.txt` lists the target Python packages. The `llama-cpp-python`
version is pinned to the CPU runtime verified in the existing Python 3.12
environment.

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

Install project dependencies in the existing virtual environment:

```powershell
.\venv\Scripts\python.exe -m pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt
```

Then run the project tests:

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
included. The Predict screen produces local suggested labels for faculty
review. In History, select a question to approve or correct its labels; the
original suggestion remains saved separately. Reviewed labels can support
retrieval of similar examples from that course. History also offers data
backup and JSON export.

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

The existing development environment uses Python 3.12.4. Python 3.11 remains
the target interpreter and still needs its own compatibility verification.

## Phase 6 — Install and verify the local model

The verified model is Qwen2.5-0.5B-Instruct Q4_K_M from the official
[Qwen GGUF repository](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF).
The repository publishes it under Apache-2.0. The model file is about 491 MB;
it is not included in Git or the application source distribution.

From PowerShell in the project folder, download only the Q4_K_M file:

```powershell
New-Item -ItemType Directory -Force .\models | Out-Null
curl.exe --fail --location --output .\models\qwen2.5-0.5b-instruct-q4_k_m.gguf "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf?download=true"
```

Verify the downloaded size and SHA-256 before using it:

```powershell
Get-Item .\models\qwen2.5-0.5b-instruct-q4_k_m.gguf | Select-Object Name,Length
Get-FileHash .\models\qwen2.5-0.5b-instruct-q4_k_m.gguf -Algorithm SHA256
```

Expected size: `491400032` bytes. Expected SHA-256:
`74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`.
The hash is the official Hugging Face LFS object digest for this exact file.

First try the prebuilt Windows CPU wheel and install the project dependencies:

```powershell
.\venv\Scripts\python.exe -m pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt
.\venv\Scripts\python.exe -c "import llama_cpp; print(llama_cpp.__version__)"
```

If the wheel download is unavailable, the runtime was also built successfully
from its PyPI source distribution using MinGW-w64 (MSYS2 UCRT64 GCC 14.2.0).
This takes several minutes and requires the MinGW compiler and make program to
be on `PATH`; CMake and the Python build dependencies are fetched by pip:

```powershell
$env:CMAKE_GENERATOR = "MinGW Makefiles"
$env:CMAKE_ARGS = "-DGGML_NATIVE=OFF"
$env:CXXFLAGS = "-D_WIN32_WINNT=0x0A00"
.\venv\Scripts\python.exe -m pip install --no-binary=llama-cpp-python --no-cache-dir llama-cpp-python==0.3.19
Remove-Item Env:CMAKE_GENERATOR,Env:CMAKE_ARGS,Env:CXXFLAGS
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Alternatively, on Python 3.12 x64, download
[`llama_cpp_python-0.3.19-cp312-cp312-win_amd64.whl`](https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.19/llama_cpp_python-0.3.19-cp312-cp312-win_amd64.whl)
in a browser and save it under `wheelhouse` in the project folder. Install it
locally, then install the remaining project dependencies:

```powershell
New-Item -ItemType Directory -Force .\wheelhouse | Out-Null
.\venv\Scripts\python.exe -m pip install .\wheelhouse\llama_cpp_python-0.3.19-cp312-cp312-win_amd64.whl
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

That wheel is for 64-bit CPython 3.12 on Windows. Do not use it with Python
3.11 or a 32-bit interpreter.

The `llama-cpp-python` package is MIT-licensed and contains native llama.cpp
components. The prebuilt Windows x64 CPU wheel does not need a separately installed
compiler or Ollama. When redistributing the runtime, include the applicable
package/native-library notices and verify the target computer has the native
runtime required by the chosen wheel/build.

### Where to configure the model later

The model configuration lives in `src/ai/llm_client.py`:

- `default_model_path()` chooses
  `models\qwen2.5-0.5b-instruct-q4_k_m.gguf` by default.
- `LLMClient(model_path=..., context_length=..., threads=...)` can override
  the model path and CPU settings in code.
- Environment variables override the defaults without changing code:
  `CO_BL_MODEL_PATH`, `CO_BL_MODEL_CONTEXT`, and `CO_BL_MODEL_THREADS`.

The client loads the model lazily, only when `chat()` is called. The desktop
application does not load the model at startup or expose model settings in its
UI. The Predict screen uses this client to suggest CO and BL labels after a
course and question are selected. You can leave the model configuration alone
unless you need to use a different model file or CPU settings.

For a one-time PowerShell session, set these before launching the app:

```powershell
$env:CO_BL_MODEL_PATH = "D:\LocalModels\qwen2.5-0.5b-instruct-q4_k_m.gguf"
$env:CO_BL_MODEL_CONTEXT = "2048"
$env:CO_BL_MODEL_THREADS = "4"
```

For persistent Windows user-level values, run this once and restart VS Code
and the app:

```powershell
[Environment]::SetEnvironmentVariable("CO_BL_MODEL_PATH", "D:\LocalModels\qwen2.5-0.5b-instruct-q4_k_m.gguf", "User")
[Environment]::SetEnvironmentVariable("CO_BL_MODEL_CONTEXT", "2048", "User")
[Environment]::SetEnvironmentVariable("CO_BL_MODEL_THREADS", "4", "User")
```

Use the full path to the `.gguf` file for `CO_BL_MODEL_PATH`. You may omit
these settings to use the default path and CPU thread count.

Run the local model test:

```powershell
.\venv\Scripts\python.exe -m pytest -q tests\test_llm_client.py
```

When the GGUF file and runtime are installed, the real-model tests load the
model and run prompts while blocking Python socket connection APIs during
inference. The Predict screen runs model loading and inference in a background
worker so the interface remains responsive. Cancellation and timeouts are
cooperative between generated tokens; llama.cpp cannot interrupt prompt
evaluation or a token currently running in native code. Run both test modules
with:

```powershell
.\venv\Scripts\python.exe -m pytest -q tests\test_llm_client.py tests\test_predictor.py
```

The inference client performs no network calls. The real-model test blocks
Python socket connection APIs while it loads and runs the model. For an
additional OS-level offline check, disconnect from the internet after
installing the model and runtime, then rerun the model test.

## Phase 9 — Quality checks and held-out evaluation

Run the automated checks with:

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m compileall -q src tests
.\venv\Scripts\python.exe -m pip check
```

The suite covers text and scanned PDFs, password-protected PDFs, syllabus
validation, course and database persistence, malformed labels, independent
`N/A` fallbacks, missing outcomes/model files, long Unicode questions, and
network-socket blocking for local inference.

To report accuracy, prepare a UTF-8 JSON Lines file from a faculty-labelled,
held-out set. Each line must contain `expected_co`, `predicted_co`,
`expected_bl`, and `predicted_bl`. Expected labels must be CO1–CO4 and
BL0–BL4; predictions may also be `N/A`. Keep examples used for prompt design
or retrieval out of the held-out set, and do not treat unreviewed model
predictions as ground truth. Questions may be omitted from this metrics file;
if included, keep the file local because it can contain private assessment
content.

```powershell
.\venv\Scripts\python.exe -m src.evaluation.evaluate_predictions `
  .\data\evaluation\held-out.jsonl `
  --output .\data\evaluation\metrics.json
```

The report contains separate CO and BL accuracy, precision, recall and F1 for
each supported label, macro-F1 over the complete label set, and `N/A`
abstention counts and rates. No faculty-labelled held-out dataset is currently
included in this repository, so the evaluator intentionally does not report
academic accuracy until one is supplied.

Measure local inference latency and process working-set memory on the target
machine with:

```powershell
.\venv\Scripts\python.exe -m scripts.benchmark_inference
```

This performs one cold-start local inference and blocks Python socket
connections. In this development environment, one run took 3.253 seconds
including model load and reached a 545,140,736-byte peak process working set
(about 520 MiB). This is a single-run observation, not a performance guarantee;
repeat it on the target computer for representative results. Python socket
blocking is not a substitute for an OS-level disconnected-network test.

## Phase 10 — Build and verify the Windows application

The release is a one-folder distribution, not a single-file executable. The
folder contains the application executable, Python/Qt/native runtime files,
classification configuration, and the verified Qwen GGUF model. Expect a
large download: this development build is about 0.78 GiB. Keep and distribute
the whole `dist\CO-BL-Predictor` folder together.

On Windows, with the project environment and verified model installed, build
and run the packaged asset/inference smoke test:

```powershell
.\scripts\build_windows.ps1
```

The script checks the model SHA-256, builds with PyInstaller, then starts the
packaged executable in a private self-test mode. The test loads the bundled model,
classification rules, and SQLite database, with writable files redirected to
a temporary user-data directory. It removes only that temporary test directory
after a successful or failed run. The ordinary application stores its
database and editable classification configuration in
`%LOCALAPPDATA%\CO-BL-Predictor`, outside the installation and extraction
directories. The packaged application checks that its bundled model exists,
is readable, and has the GGUF signature before showing the main window.

For distribution, archive or install the complete output folder; do not copy
only the `.exe`. The PyInstaller folder already contains Python and the native
llama.cpp/Qt/PDF libraries, so the target does not need a separate Python or
Ollama installation. The tested model is approximately 491 MB and subject to
its published Apache-2.0 license. PyMuPDF distribution requires review of its
AGPL-3.0 or commercial licensing terms. Review `THIRD_PARTY_NOTICES.md` and
the licenses for all bundled dependencies before redistribution.

The build and smoke test run on the current Windows development computer.
They do not replace testing on a separate clean Windows machine, an
OS-level-disconnected network, or a target computer without Python installed.
Do those release checks before publishing a distributable.

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
- Phase 6 — Local Qwen/llama.cpp integration: complete. The official Q4_K_M
  model is downloaded and checksum-verified; the CPU runtime is installed;
  configuration, failure handling, background execution, cancellation, and
  timeout are implemented and covered. Real-model prompts passed with Python
  socket connections blocked under Python 3.12.
- Phase 7 — Core CO/BL prediction: separate structured prompts, strict label
  validation with independent `N/A` fallbacks, reviewed syllabus context,
  SQLite prediction metadata, and the asynchronous Predict UI are implemented
  and tested.
- Phase 8 — Approved-example retrieval and review: faculty can approve or
  correct saved suggestions in History; original predictions remain unchanged.
  Same-course reviewed examples are ranked with TF-IDF and cosine similarity.
  Up to three examples with similarity of at least 0.15 enter prediction
  context. The retrieval index is rebuilt from current reviews per prediction,
  so new corrections are immediately available. This supplies context; it does
  not train or fine-tune the local model.
- Phase 9 — Automated tests, held-out CO/BL metric reporting, and a local
  latency/memory benchmark are implemented. The full test suite and local
  benchmark passed. A real faculty-labelled held-out set and OS-level
  disconnected-network test are still required before academic accuracy or
  fully offline operation can be certified.
- Phase 10 — Windows one-folder PyInstaller build (tested under Python 3.12.4),
  bundled verified model,
  configuration, application icon/version metadata, per-user writable
  configuration/database paths, startup model validation, and packaged
  inference smoke test are implemented. Clean-machine and OS-level offline
  acceptance tests remain outstanding.
