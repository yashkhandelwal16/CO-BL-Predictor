Bhai 🙌, ab hum technology stack final kar dete hain aur VS Code Copilot ko phase-by-phase development karwayenge.

Your project will be a local-first CO & BL Predictor for Windows, with the Qwen model bundled into the application and `llama.cpp` acting as the inference engine. The application will read course syllabus PDFs, extract unit-wise topics and COs, and predict labels without answering exam questions.

# 1. Finalized technology stack

| Component                 | Technology                              |
| ------------------------- | --------------------------------------- |
| Programming language      | Python 3.11                             |
| Desktop GUI               | PySide6                                 |
| AI model                  | Qwen2.5-0.5B-Instruct                   |
| Model format              | GGUF, initially Q4_K_M                  |
| Inference middleware      | `llama.cpp` through `llama-cpp-python`  |
| PDF text extraction       | PyMuPDF                                 |
| Scanned PDF support       | OCR as a later phase                    |
| Syllabus storage          | JSON                                    |
| Application database      | SQLite                                  |
| NLP and similarity search | scikit-learn, TF-IDF, cosine similarity |
| Packaging                 | PyInstaller                             |
| Testing                   | pytest                                  |
| Version control           | Git + GitHub                            |

Important architecture decision: use `llama-cpp-python` as the Python binding to `llama.cpp`. This avoids requiring a separate Ollama installation or background server. The model file remains a separate bundled asset inside the packaged application.

The final output must be exactly:

```
CO level - CO1
BL level - BL1
```

If the evidence is insufficient, return `N/A` for the uncertain field. Never generate an answer to the question.

# 2. Master development task list for VS Code Copilot

Use this as your project's master checklist. Complete the phases in order; don't ask Copilot to generate the entire application in one prompt.

## Project milestones

0 of 11 phases marked complete

Reset

Phase 0 — Workspace and requirements

Inspect repository, define architecture and project rules.

Phase 1 — Python environment and GUI

Create the runnable PySide6 desktop interface.

Phase 2 — Course management

Create, edit and manage courses and CO1–CO5.

Phase 3 — PDF extraction and JSON

Parse syllabus PDFs into unit-wise structured data.

Phase 4 — BL configuration

Add BL0–BL5 levels and configurable keywords.

Phase 5 — SQLite persistence

Store courses, syllabi, predictions and corrections.

Phase 6 — Local Qwen inference

Integrate GGUF model through llama-cpp-python.

Phase 7 — CO and BL prediction

Build the prediction pipeline, output validation and N/A rules.

Phase 8 — NLP and approved examples

Add similarity retrieval and faculty feedback.

Phase 9 — Testing and quality

Test correctness, performance, privacy and failure cases.

Phase 10 — Windows packaging

Build and test a distributable executable with model assets.

## Phase 0 — Workspace setup and architecture

Goal: establish a clean, maintainable project before implementation.

- Open the project folder in VS Code.
- Ask Copilot to inspect existing files before changing anything.
- Create or review `README.md`.
- Create `requirements.txt`.
- Create `.gitignore`.
- Create `.github/copilot-instructions.md`.
- Define the folder structure below.
- Document the local-only architecture.
- Confirm that no cloud API is required.
- Define the output format and the `N/A` rules.
- Define the supported CO labels: CO1–CO5.
- Define the supported BL labels: BL0–BL5.
- Record the model repository, exact GGUF filename, license and checksum once selected.

Suggested folder structure:

```
CO-BL-Predictor/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── .github/
│   └── copilot-instructions.md
├── config/
│   └── bloom_levels.json
├── models/
│   └── qwen2.5-0.5b-instruct-q4_k_m.gguf
├── data/
│   ├── syllabi/
│   ├── parsed_json/
│   └── exports/
├── src/
│   ├── ui/
│   ├── models/
│   ├── services/
│   ├── syllabus/
│   ├── prediction/
│   ├── ai/
│   ├── nlp/
│   ├── database/
│   └── utils/
├── tests/
├── scripts/
└── packaging/
```

Copilot must use the actual downloaded model filename rather than assuming the illustrative filename above.

Acceptance criteria: the workspace has a documented architecture, a Git-safe configuration, and no model inference or GUI implementation has been mixed into one large file.

## Phase 1 — Python environment and GUI

Goal: get a basic desktop application running before integrating AI.

### Tasks

- Create a Python virtual environment.
- Install PySide6.
- Install pytest.
- Create `main.py`.
- Create `src/ui/main_window.py`.
- Create a main window with a sensible minimum size.
- Add a sidebar or navigation tabs.
- Add pages for Dashboard, Courses, Syllabus, Predict, History and Settings.
- Apply an aqua-blue and pale-white theme.
- Add form validation and readable error messages.
- Add loading indicators for operations that may take time.
- Keep long-running operations off the GUI thread.
- Verify that the application starts and closes cleanly.

### GUI screens

Dashboard

Course count, recent predictions and quick actions.

Courses

Create, edit, search and select courses.

Syllabus

Upload PDFs and review extracted units and COs.

Predict

Enter a question and display only CO and BL labels.

Acceptance criteria: `python main.py` launches a functioning desktop GUI with navigation. Mock data is acceptable at this stage; fake AI predictions must be clearly marked as placeholders.

## Phase 2 — Course management and CO input

Goal: let users create courses without hardcoding their subjects.

### Tasks

- Build a Create Course form.
- Add Course Name.
- Add Course Code.
- Add syllabus PDF selection.
- Add four separate CO input fields: CO1, CO2, CO3 and CO4.
- Populate CO fields from PDF extraction when available.
- Allow manual correction of every CO.
- Support editing and deleting a course.
- Add search by course name and code.
- Validate required fields.
- Prevent accidental duplicate course codes.
- Show a confirmation before deleting course data.

Acceptance criteria: a user can create two different courses, each with its own syllabus and four independent CO descriptions. Changing one course must not modify another course.

## Phase 3 — Syllabus PDF processing

Goal: turn an uploaded syllabus into structured, reviewable JSON.

### Tasks

- Create `src/syllabus/pdf_extractor.py`.
- Use PyMuPDF to extract text from text-based PDFs.
- Preserve page numbers for debugging and traceability.
- Detect empty or unreadable documents.
- Detect likely unit headings.
- Extract topics under each unit.
- Detect course outcomes when present.
- Extract course name and code when present.
- Save parsed data to a JSON file.
- Create a JSON schema or equivalent validator.
- Display the parsed syllabus in an editable preview.
- Flag uncertain or missing units and outcomes for manual review.
- Preserve the original PDF.
- Add OCR as a separate fallback for scanned PDFs.
- Never silently invent missing syllabus content.

### Example JSON output

```
{
  "course_name": "Example Course",
  "course_code": "CS301",
  "units": [
    {
      "unit_number": 1,
      "unit_title": "Introduction",
      "topics": [
        "Topic A",
        "Topic B"
      ],
      "source_pages": [1, 2]
    }
  ],
  "course_outcomes": [
    {
      "id": "CO1",
      "description": "Understand fundamental concepts",
      "source_pages": [2]
    }
  ],
  "review_status": "pending"
}
```

The sample content above is illustrative. Real data must be extracted from the user's PDF or entered manually.

Acceptance criteria: uploading a test PDF produces valid JSON, unit headings and topics can be reviewed, and the user can correct extraction errors before predictions begin.

## Phase 4 — Bloom's Taxonomy configuration

Goal: implement BL0–BL5 as predefined but configurable rules.

### Tasks

- Create `config/bloom_levels.json`.
- Define BL0 as a project-specific fallback or unclassifiable state.
- Define BL1, BL2, BL3, BL4 and BL5 (Evaluate).
- Add a list of example keywords for each level.
- Validate the configuration when the app starts.
- Add a Settings screen to edit definitions and keywords.
- Support saving and restoring default rules.
- Build a keyword-matching service.
- Support case-insensitive matching.
- Avoid matching partial words accidentally.
- Handle questions containing multiple cognitive verbs.
- Keep keyword evidence separate from the final predicted label.

Suggested initial configuration:

```
{
  "levels": [
    {
      "id": "BL0",
      "name": "Unclassifiable",
      "keywords": []
    },
    {
      "id": "BL1",
      "name": "Remember",
      "keywords": [
        "define",
        "list",
        "state",
        "identify",
        "recall",
        "name"
      ]
    },
    {
      "id": "BL2",
      "name": "Understand",
      "keywords": [
        "explain",
        "describe",
        "summarize",
        "classify",
        "interpret"
      ]
    },
    {
      "id": "BL3",
      "name": "Apply",
      "keywords": [
        "calculate",
        "solve",
        "determine",
        "compute",
        "demonstrate"
      ]
    },
    {
      "id": "BL4",
      "name": "Analyze",
      "keywords": [
        "compare",
        "differentiate",
        "analyze",
        "examine",
        "investigate"
      ]
    }
  ]
}
```

Acceptance criteria: changing a keyword in Settings changes the rule engine's evidence without requiring code changes. A keyword match alone must not force a final classification.

## Phase 5 — SQLite database

Goal: persist all important data across application restarts.

### Tasks

- Create `src/database/db_manager.py`.
- Create and initialize the SQLite database.
- Define a `courses` table.
- Define a `syllabus_versions` table.
- Define a `course_outcomes` table.
- Define a `questions` table.
- Define a `predictions` table.
- Define an `approved_labels` or review table.
- Store timestamps for creation and modification.
- Store the model name and version with each prediction.
- Store the syllabus version used for each prediction.
- Use parameterized SQL queries.
- Add database transactions and error handling.
- Implement database backup and export.
- Keep user data in a writable application-data directory, not inside a read-only installation folder.

Acceptance criteria: create a course, restart the application, and verify that the course, CO descriptions and syllabus metadata remain available.

## Phase 6 — Integrate Qwen2.5-0.5B-Instruct locally

Goal: run model inference without Ollama or a cloud API.

### Tasks

- Download a compatible Qwen2.5-0.5B-Instruct GGUF model.
- Start with Q4_K_M quantization, subject to availability and compatibility.
- Verify the exact model license and redistribution terms.
- Record the model's source URL and checksum.
- Install a compatible `llama-cpp-python` build.
- Create `src/ai/llm_client.py`.
- Load the model from a configurable local path.
- Configure context length and CPU thread usage.
- Test a basic prompt.
- Implement inference timeouts or cancellation where feasible.
- Handle missing, corrupt or incompatible model files.
- Ensure the GUI remains responsive during inference.
- Test with internet disabled after all dependencies and model assets have been installed.
- Document the runtime's redistribution and native-library requirements.

### Model loading design

```
from pathlib import Pathfrom llama_cpp import Llamadef load_model(model_path: str) -> Llama:    path = Path(model_path)    if not path.is_file():        raise FileNotFoundError(            f"Local model not found: {path}"        )    return Llama(        model_path=str(path),        n_ctx=2048,        verbose=False,    )
```

This is a minimal example for Copilot to build on, not the complete production implementation. Copilot must test compatibility with the exact GGUF file and installed binding version.

Acceptance criteria: the local model loads successfully, classifies a small set of test prompts, and produces no network requests during inference.

## Phase 7 — Core CO and BL prediction engine

Goal: produce only the required labels.

### Tasks

- Create `src/prediction/predictor.py`.
- Create a prompt template for CO classification.
- Create a prompt template for BL classification.
- Include the selected course's CO descriptions.
- Retrieve relevant syllabus units.
- Include BL definitions and keywords.
- Tell the model explicitly not to answer the question.
- Require a structured response.
- Validate the returned CO against CO1–CO5.
- Validate the returned BL against BL0–BL5.
- Reject explanations, invented labels and malformed output.
- Allow independent CO and BL fallback results.
- Return `N/A` for a field if the evidence is insufficient.
- Record the prediction and supporting metadata in SQLite.
- Display only the two required output lines.
- Test empty input, ambiguous questions and unrelated questions.

### Required external output

```
CO level - CO1
BL level - BL1
```

Or, if only the BL result is uncertain:

```
CO level - CO1
BL level - N/A
```

Or, if both are uncertain:

```
CO level - N/A
BL level - N/A
```

Acceptance criteria: the output formatter emits only the two lines, and no generated answer or additional explanation appears in the result field.

## Phase 8 — NLP similarity search and approved examples

Goal: improve consistency by reusing previously reviewed examples.

### Tasks

- Create `src/nlp/text_preprocessor.py`.
- Implement TF-IDF vectorization.
- Implement cosine similarity search.
- Index approved questions separately for each course.
- Retrieve the most similar approved questions.
- Include only relevant examples in the model's context.
- Keep examples associated with their approved CO and BL labels.
- Add a faculty review/correction workflow.
- Mark corrected labels as reviewed rather than silently overwriting the original prediction.
- Rebuild the similarity index when approved examples change.
- Avoid using unreviewed predictions as ground truth.
- Add a minimum similarity threshold.
- Return `N/A` if evidence remains inadequate.

Acceptance criteria: approved examples can be retrieved for similar questions from the same course, while unrelated courses and unreviewed labels are excluded.

Remember: retrieving an approved question is not the same as training the Qwen model. This phase improves context and consistency; actual model fine-tuning would be a separate project.

## Phase 9 — Testing and quality assurance

Goal: verify that the application is reliable before packaging.

### Tasks

- Add tests for PDF extraction.
- Add tests for JSON validation.
- Add tests for course management.
- Add tests for database persistence.
- Add tests for BL keyword matching.
- Add tests for output validation.
- Add tests for `N/A` handling.
- Add tests for missing model files.
- Add tests for malformed model output.
- Test long questions and unusual characters.
- Test a course with missing outcomes.
- Test scanned and password-protected PDFs.
- Measure inference time and peak memory usage.
- Test with internet disconnected.
- Build a labelled academic test set.
- Measure CO accuracy and BL accuracy separately.
- Measure per-class precision, recall and macro-F1.
- Record abstention rates.
- Fix all critical failures before release.

Acceptance criteria: tests pass, invalid labels are rejected, and accuracy results are measured on data that was not used to design or tune the system.

## Phase 10 — Package the Windows `.exe`

Goal: distribute the desktop application with the model included.

### Tasks

- Install and configure PyInstaller.
- Create a packaging script or `.spec` file.
- Include the model GGUF as a bundled asset.
- Include the required native inference libraries.
- Include GUI and PDF dependencies.
- Ensure the app resolves the model path correctly in development and packaged modes.
- Store SQLite and user files outside the temporary extraction directory.
- Add startup checks for missing or unreadable model assets.
- Add application icon and version metadata.
- Review third-party licenses and model redistribution terms.
- Build a clean Windows release.
- Test on a machine without Python installed.
- Test on a machine without Ollama installed.
- Test offline inference.
- Verify that course data survives application restarts and updates.
- Measure final installer size.
- Document installation, model attribution and troubleshooting.

### Packaging strategy

For a bundled model, Copilot should investigate PyInstaller's `--add-data` and binary inclusion options, and the correct path resolution for packaged applications. A one-file PyInstaller build may extract the model to a temporary directory at startup; that can add launch overhead and consume temporary disk space.

A folder-based distribution or installer containing the `.exe`, native libraries and model may be more practical than a literal single-file executable. You can still distribute it as one downloadable installer.

Acceptance criteria: the packaged application runs on a clean Windows test machine, does not require a separate Ollama installation, can load the bundled model, and works offline.

# 3. Copilot master instruction file

Create `.github/copilot-instructions.md` and paste the following into it. This helps Copilot follow your project requirements across tasks.

# CO & BL Predictor — Copilot Project Instructions

## Project objective

Build a Windows desktop application that predicts Course Outcome labels (CO1–CO5) and Bloom's Taxonomy labels (BL0–BL5) for academic exam questions.

The application must not answer exam questions. It must return only the requested CO and BL labels, or N/A when evidence is insufficient.

## Fixed technology stack

- Python 3.11
- PySide6 desktop GUI
- Qwen2.5-0.5B-Instruct in GGUF format
- llama.cpp through llama-cpp-python
- PyMuPDF for PDF extraction
- SQLite for persistent data
- JSON for structured syllabus and BL configuration
- scikit-learn for TF-IDF and cosine similarity
- pytest for testing
- PyInstaller for Windows packaging

Do not add Ollama or cloud AI APIs as required runtime dependencies.

## Core functional requirements

1. Users can create and manage courses using course name and course code.
2. Users can upload syllabus PDFs.
3. The application extracts syllabus text, units, topics and available course outcomes into structured JSON.
4. Users can review and correct extracted information.
5. Each course may have separate CO1–CO5 fields.
6. Custom cognitive levels BL0–BL5 and their keywords are configurable; BL5 means Evaluate and definitions remain subject to institutional approval.
7. Users can submit academic questions for local prediction.
8. Predictions use the selected course's verified CO descriptions and relevant syllabus topics.
9. Approved historical examples may be retrieved to support prediction.
10. Predictions and faculty corrections are stored separately.
11. The application works offline once installed with all required dependencies and model assets.
12. The final Windows distribution includes the Qwen model and required inference runtime.

## Required output format

Successful prediction:

CO level - CO1\
BL level - BL1

Uncertain CO:

CO level - N/A\
BL level - BL2

Uncertain BL:

CO level - CO3\
BL level - N/A

The GUI prediction result must contain only these two lines. Do not include explanations, reasoning, answers, confidence text or Markdown in the displayed result.

Internally, use a validated structured representation such as:

{"co": "CO1", "bl": "BL1"}

Validate every returned label in Python. Never trust model output without validation.

## Prediction policy

- Never answer the user's exam question.
- Never invent a course outcome.
- Never use a CO from a different course.
- Never infer missing syllabus information as fact.
- Use BL keywords as supporting evidence, not as the only classification rule.
- Consider the complete wording and cognitive intent of the question.
- Return N/A for a field when there is insufficient evidence.
- Treat CO and BL uncertainty independently.
- Do not assume the model's self-reported confidence is calibrated.
- Do not silently convert malformed output into a valid prediction.
- Record model predictions separately from faculty-approved labels.

## Architecture rules

- Keep UI code separate from business logic.
- Keep PDF extraction separate from JSON validation.
- Keep model inference separate from prediction rules.
- Keep prediction formatting and label validation in dedicated modules.
- Keep database access in the database layer.
- Do not place the complete application in one Python file.
- Do not perform long-running inference directly on the GUI thread.
- Use parameterized SQL queries.
- Use logging and clear user-facing errors.
- Use type hints and docstrings for public functions.
- Use pathlib for filesystem paths.
- Avoid hardcoded absolute paths.
- Keep model and user-data paths configurable.
- Store writable user data outside the application installation directory.
- Do not make network requests during prediction.

## Model integration

Use llama-cpp-python with a compatible Qwen2.5-0.5B-Instruct GGUF file.

Verify the actual model filename, file existence, runtime compatibility, license and checksum. Do not fabricate model metadata.

Make model loading configurable and handle missing or corrupt model files gracefully.

Use a context length suitable for the selected model and available memory. Keep inference parameters configurable and test them.

Do not claim that the model is accurate without evaluation on a labelled dataset.

## PDF and syllabus rules

Preserve the original PDF. Record extraction failures and source-page references where possible.

Allow users to review and edit extracted data before using it for prediction.

Do not silently fabricate units, topics, CO descriptions or course metadata.

Handle empty, scanned, malformed and unreadable PDFs gracefully.

## Database rules

Persist courses, syllabus versions, CO descriptions, questions, predictions and approved labels.

Record timestamps and the relevant model and syllabus version.

Use transactions and parameterized SQL.

Do not treat unreviewed model predictions as ground truth.

## Privacy and packaging

The app must operate locally without requiring a cloud API.

Never upload course files, exam questions or user data.

Never commit private data, model files or unnecessary generated files to Git.

Before redistributing the model, verify its license and applicable attribution requirements.

The packaged Windows application must include the model and compatible inference dependencies. It must not require a separate Ollama installation.

Use PyInstaller and test the release on a clean Windows machine without Python installed.

## Testing

Write pytest tests for PDF extraction, syllabus validation, database operations, keyword matching, prediction output validation and N/A handling.

Mock model inference in unit tests where appropriate.

Keep separate integration tests for actual local model inference.

Test empty input, unrelated questions, ambiguous questions, missing outcomes, invalid model responses and missing model files.

Do not mark a phase complete until its tests pass and its acceptance criteria are met.

## Copilot workflow

Implement one phase at a time.

Before editing files:

1. Inspect the existing workspace.
2. Identify files that need to be created or changed.
3. Explain the planned changes briefly.
4. Preserve existing working functionality.

After implementing each phase:

1. List changed files.
2. Explain how the code works in beginner-friendly language.
3. Give exact installation and run commands.
4. Run relevant tests or provide the exact command to run them.
5. Report errors honestly.
6. List any remaining limitations.
7. Stop and wait for confirmation before beginning the next phase.

Never claim code has been executed or tested unless it actually has been.

# 4. Copilot prompt for starting development

After saving the instruction file, open Copilot Chat in VS Code and send this first prompt:

Read `.github/copilot-instructions.md` completely and inspect the current workspace.

We are beginning Phase 0 of the CO & BL Predictor project.

My application must be a local-first Windows desktop app using Python 3.11, PySide6, Qwen2.5-0.5B-Instruct GGUF, llama.cpp through llama-cpp-python, PyMuPDF, SQLite, JSON, scikit-learn, pytest and PyInstaller.

The final Windows distribution must include the model and required inference runtime. It must not require Ollama, a cloud API, or a separate model download at runtime.

First, audit the current workspace. Do not rewrite the application or generate every phase at once.

Please:

1. Explain the proposed architecture in simple English.
2. Identify what already exists and what is missing.
3. Check whether Python and the virtual environment are configured, if the terminal tools allow it.
4. Propose the exact folder structure.
5. Identify compatibility and packaging risks for llama-cpp-python and PyInstaller on Windows.
6. Confirm the required output format and N/A behavior.
7. Produce a Phase 0 checklist with commands I can run in VS Code.
8. Tell me exactly what to do next.

Do not invent execution results. Do not begin Phase 1 until the Phase 0 audit and checklist are complete.

## 5. The development rule that will save you the most time

Never ask Copilot to implement all 10 phases in one request.

Follow this cycle for every phase:

1. Ask Copilot to plan the phase.
2. Review the files it intends to change.
3. Ask it to implement only that phase.
4. Run the application or tests.
5. Share any error with Copilot and fix it.
6. Confirm the acceptance criteria.
7. Move to the next phase.

Start with Phase 0, then complete the GUI and syllabus workflow before integrating the Qwen model. This will make debugging significantly easier and help you understand what each part of the application actually does.