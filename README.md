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
to public search services. The integrated `ResearchEngine` sanitizes this
query before sending anything to DDGS, as described below.

## Optional CO-specific web research

`ResearchEngine` combines the syllabus-aware ranking, DDGS search, and webpage
extraction services. It returns structured evidence grouped by candidate CO,
including the search query, title, URL, snippet, extracted content when
available, publication/retrieval dates, provider, relevance score, source
quality hint, extraction status, and a safe internal extraction diagnostic.
Source-quality hints prioritize likely
university, academic, educational, or official-documentation domains; they do
not verify that a source is accurate.

The engine applies bounded query/result/page limits, filters low-relevance
results, removes duplicate URLs within candidate results, caches successful
search responses in memory, and reuses the extraction service's page cache.
When the same URL is relevant to multiple candidate COs, each CO association
is retained while the page is fetched only once per research run. DDGS errors
are reported in the result status; a failed page keeps its search snippet.
`ResearchWorker` runs the workflow off the GUI thread. Cancellation is checked
between searches and during extraction; an in-progress DDGS call ends on its
configured timeout.

For privacy, the external DDGS query contains only caller-supplied technical
concepts that locally match the candidate outcome, plus the cognitive action
and generic academic terms. Supply only concepts suitable for public search.
The question text, course identity, official CO description, and syllabus
topics are used locally for ranking and are excluded from the external query.
If no such concepts are supplied, that candidate is not searched. Web evidence
is research context only and never assigns an institutional CO or changes
local prediction.

```python
from src.research import ResearchEngine

research = ResearchEngine(max_queries=3, results_per_query=5)
evidence = research.research(
    academic_question=question,
    course_name=course.name,
    course_code=course.code,
    course_outcomes=reviewed_course_outcomes,
    unit_topics=reviewed_unit_topics,
    relevant_technical_concepts=technical_concepts,
    detected_cognitive_action=cognitive_action,
)
for source in evidence.sources:
    print(source.candidate_co_id, source.title, source.url)
    print(source.extracted_content or source.search_snippet)
```

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
