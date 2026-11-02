# CO & BL Predictor

A Windows desktop app that helps faculty map exam questions to course outcomes
(CO1–CO5) and cognitive levels (BL0–BL5). Predictions run locally and are
suggestions for faculty review, not final decisions.

## Features

- Manage courses and their course outcomes.
- Import, review, and save syllabus PDFs.
- Predict CO and BL labels with a local Qwen model.
- Review predictions and browse question history.
- Keep courses and prediction data in a local SQLite database.

## Run

Requires Python and the local Qwen GGUF model. Place the model at
`models\qwen2.5-0.5b-instruct-q4_k_m.gguf`, then install dependencies and start
the app:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py
```

The app does not require a cloud AI service. Model and application data are
stored locally.

## Optional web search service

The standalone web search service uses DDGS for public text search without an
API key. Trafilatura can extract readable text from a selected public webpage.
Search and page extraction are optional services; they are not connected to
question prediction. Search queries are sent to public search backends, so only
use public research topics; never pass private course data, syllabus content,
or examination questions.

Install the declared dependencies with the normal requirements command above.
The service can be used independently:

```python
from src.research import WebSearchService

service = WebSearchService(max_results=5, backend="auto", region="us-en")
results = service.search("Java inheritance and polymorphism concepts")
for result in results:
    print(result.title, result.url, result.snippet, result.source)
```

DDGS backends and region, safe-search, time-limit, result-count, and timeout
options can be configured through `WebSearchService`. Searches require an
internet connection and may be limited or blocked by search providers. No
automatic retries are made. Webpage extraction is an explicit, separate call
and only permits public HTTP(S) destinations.

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
