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
automatic retries are made. Webpage extraction is a separate, optional
operation:

```python
from src.research import WebContentExtractionService

extractor = WebContentExtractionService(max_pages=3)
pages = extractor.extract_results(results)
for page in pages:
    print(page.title, page.url, page.extraction_status)
    if page.extraction_status in {"success", "cached"}:
        print(page.content)
    else:
        print(page.snippet)  # Original search-result fallback
```

Extraction checks `robots.txt`, accepts only public HTTP(S) destinations, and
revalidates every redirect. Connections are pinned to a resolved public IP
while HTTPS hostname and certificate verification remain enabled. Pages are
limited by count, timeout, response size, and an in-memory successful-result
cache. Failed, blocked, invalid, or unreadable pages retain their original
search title, URL, and snippet with a status and internal diagnostic. The
standalone `WebContentExtractionWorker` can run this work away from the GUI
thread. Neither search nor extraction is connected to question prediction.

## Optional syllabus-aware query generation

`SyllabusAwareQueryGenerator` uses the existing local TF-IDF/cosine helper,
concept overlap, and reviewed unit topics to rank only the course outcomes
provided by its caller. It returns ranked evidence, plausible CO candidates,
candidate-specific search queries, and a reason for each query. It does not
assign outcomes from web results, change official CO descriptions, call a
search provider, or alter prediction behavior.

```python
from src.research import SyllabusAwareQueryGenerator

analysis = SyllabusAwareQueryGenerator().generate(
    academic_question=question,
    course_name=course.name,
    course_code=course.code,
    course_outcomes=reviewed_course_outcomes,
    unit_topics=reviewed_unit_topics,
    relevant_technical_concepts=technical_concepts,
    detected_cognitive_action=cognitive_action,
)
for candidate_query in analysis.generated_queries:
    print(candidate_query.co_id, candidate_query.query, candidate_query.reason)
```

The generated query includes the academic question. Review it before any
external search; do not send private course material or examination questions
to public search services.

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
