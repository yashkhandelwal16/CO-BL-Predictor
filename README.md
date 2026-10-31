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

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest -q
```
