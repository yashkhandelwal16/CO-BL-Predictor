# CO & BL Predictor

A Windows desktop app that helps faculty map exam questions to course outcomes
(CO1–CO5) and Revised Bloom's Taxonomy levels (BL1–BL6). Predictions run
locally and are suggestions for faculty review, not final decisions. Older
custom BL0–BL5 history remains distinguishable from the revised labels.

## Features

- Manage courses and their course outcomes.
- Import, review, and save syllabus PDFs.
- Predict CO and BL labels with a local Qwen model.
- Show an independent DDGS-based Web Search suggestion beside the local result.
- Review predictions and browse question history.
- Keep courses and prediction data in a local SQLite database.

The local model prompt and the independent Web Search classifier share the
Revised Bloom's Taxonomy knowledge base in
[`src/prediction/bloom_taxonomy.py`](./src/prediction/bloom_taxonomy.py).
It defines BL1 Remember, BL2 Understand, BL3 Apply, BL4 Analyze, BL5 Evaluate,
and BL6 Create. Classification considers the full question and expected
student response, not an action verb, length, difficulty, or programming
format alone. When the evidence does not support a level, the result is `N/A`.

Web Search sends only short technical concepts to public search, not the
question text, course identity, official CO descriptions, or syllabus text.
Clear web evidence is preferred; when it is missing or ambiguous, the panel
shows the best syllabus-based CO match and labels it as a faculty-review
fallback.

## Run

Requires Python and the local Qwen GGUF model. Place the model at
`models\qwen2.5-0.5b-instruct-q4_k_m.gguf`, then install dependencies and start
the app:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py
```

The app does not require a cloud AI service. Model and application data are
stored locally. This MVP version keeps its data separate from data created by
newer versions of the app.

## Build a portable Windows executable

With the verified model at `models\qwen2.5-0.5b-instruct-q4_k_m.gguf` and the
project build dependencies installed, run:

```powershell
.\scripts\build_windows.ps1
```

The standalone executable is written to
`dist\CO-BL-Predictor.exe`. It bundles the model and runtime dependencies; end
users do not need to install Python or application packages. The one-file app
extracts its bundled files to a temporary directory when it starts, so allow
extra disk space for the executable and its temporary runtime files. This is a
Windows-only build.

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest -q
```
