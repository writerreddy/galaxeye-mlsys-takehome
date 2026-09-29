# GalaxEye Backend Engineer — ML Systems Take-Home

Offline satellite tile land-use classification service: one endpoint takes a
tile image, classifies it locally (no internet, no hosted APIs), and stores
the result for later querying.

## Overview

This repository is a working thin slice of the system described in
[`DESIGN.md`](DESIGN.md), plus the written answers in [`PART3.md`](PART3.md).
It classifies a satellite tile into one of 7 land-use classes (`AnnualCrop`,
`Forest`, `Highway`, `Industrial`, `Residential`, `River`, `SeaLake`) using a
locally trained classical ML model, and stores every result in SQLite.

## Architecture

```
Client --> POST /tiles/classify --> validate --> preprocess --> local model
                                                                     |
                                                                     v
                                                   confidence policy (accepted/uncertain)
                                                                     |
                                                                     v
                                                              SQLite (persisted)
                                                                     |
                                                                     v
                                               GET /results, GET /results/{tile_id}
```

See [`DESIGN.md`](DESIGN.md) for the full architecture discussion, storage
model, and trade-offs.

## Repository Structure

```
galaxeye-takehome/
├── app/
│   ├── main.py                    FastAPI app, startup/lifespan
│   ├── api/routes.py              HTTP endpoints
│   ├── core/config.py             Settings (env-configurable)
│   ├── core/logging.py            Logging setup
│   ├── db/database.py             SQLAlchemy engine/session
│   ├── db/models.py               ClassificationResult ORM model
│   ├── ml/preprocessing.py        Image validation + feature extraction
│   ├── ml/classifier.py           Loads the trained model, runs inference
│   ├── schemas/classification.py  Pydantic request/response schemas
│   └── services/classification_service.py   Orchestrates validate -> classify -> store
├── scripts/
│   ├── train.py                   Trains the model from candidate_tiles/, saves the artifact
│   └── evaluate.py                Evaluates the saved model against eval_set/
├── tests/                         pytest suite (22 tests)
├── models/
│   └── land_use_classifier.joblib Pre-trained model artifact (checked in — see below)
├── data/
│   └── be-mlsys-assignment-dataset/   The supplied dataset (candidate_tiles/, eval_set/, eval_labels.csv)
├── DESIGN.md                      Part 1: design note
├── PART3.md                       Part 3: written answers
└── REVIEW.md                      Self-review: requirement coverage + anticipated interview Q&A
```

## Requirements

- Python 3.11+
- No internet access needed at runtime (only for the initial `pip install`)

## Installation

```bash
python -m venv .venv
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

## Model Preparation

A trained model artifact is already checked in at
`models/land_use_classifier.joblib`, so the API works out of the box. To
retrain it from scratch (deterministic — same seed, same result):

```bash
python scripts/train.py
```

This loads every image under `data/be-mlsys-assignment-dataset/candidate_tiles/`
(1050 labeled tiles across 7 classes), extracts features, does an 80/20
stratified train/validation split, trains a `RandomForestClassifier`, prints
a validation classification report, and overwrites
`models/land_use_classifier.joblib`.

## Running the API

```bash
uvicorn app.main:app --reload
```

The API is now at `http://localhost:8000`. Interactive docs at
`http://localhost:8000/docs`.

## API Usage

```bash
curl -X POST \
  -F "file=@data/be-mlsys-assignment-dataset/eval_set/tile_001.png" \
  http://localhost:8000/tiles/classify
```

Other endpoints:

```bash
curl http://localhost:8000/health

curl "http://localhost:8000/results?predicted_class=Forest&min_confidence=0.7"

curl http://localhost:8000/results/<tile_id>
```

## Example Response

Accepted (confident) prediction:
```json
{
  "tile_id": "5442da66-6da6-47c2-bdeb-d27a0cf5e568",
  "predicted_class": "Forest",
  "confidence": 0.995,
  "status": "accepted",
  "model_version": "v1",
  "processing_time_ms": 18.4
}
```

Uncertain (low-confidence) prediction:
```json
{
  "tile_id": "...",
  "predicted_class": "River",
  "confidence": 0.42,
  "status": "uncertain",
  "model_version": "v1",
  "processing_time_ms": 12.1
}
```

Both examples above are real output from this codebase, run against the
supplied dataset — not illustrative placeholders.

## Running Tests

```bash
pytest
```

22 tests covering health, input validation, database persistence,
confidence/status behavior, and one true integration test that exercises
the real trained model end-to-end (upload → classify → store → retrieve).
The rest use a mocked classifier so they're fast and don't depend on model
accuracy.

## Running Evaluation

```bash
python scripts/evaluate.py
```

Evaluates the saved model against `eval_set/` + `eval_labels.csv` — a
held-out set the model was never trained on. Prints overall accuracy,
per-class precision/recall/F1, a confusion matrix, and a breakdown of
accuracy at/above vs. below the confidence threshold. Also saves
`models/confusion_matrix.png` if matplotlib is installed.

**Actual measured results on the held-out eval set (210 tiles, never used
in training):**

- Overall accuracy: **85.2%**
- Forest, SeaLake, Industrial, Residential: 93-98% F1
- Highway, River: 61-67% F1 (these two get confused with each other —
  both are long, thin, linear features against variable backgrounds)
- At the default confidence threshold (0.60): predictions marked
  `accepted` (152/210) are correct **97.4%** of the time; predictions
  marked `uncertain` (58/210) are correct only **53.4%** of the time

These numbers are not cherry-picked or claimed without measurement — they're
exactly what `scripts/evaluate.py` prints when you run it. See `PART3.md`
Q1 for why "85% accuracy" is the wrong single number to optimize for here,
and why the confidence split matters more.

## Offline Guarantee

No code path in `app/` makes a network call. The model is loaded from a
local file (`models/land_use_classifier.joblib`) at startup and never
re-trained or re-downloaded at request time. All dependencies are installed
once, up front, from `requirements.txt`.

## Configuration

All settings are in `app/core/config.py`, overridable via environment
variables prefixed `GALAXEYE_`:

| Variable | Default | Meaning |
|---|---|---|
| `GALAXEYE_CONFIDENCE_THRESHOLD` | `0.60` | Predictions below this are marked `uncertain` |
| `GALAXEYE_DATABASE_URL` | `sqlite:///data/app.db` | SQLAlchemy database URL |
| `GALAXEYE_MODEL_PATH` | `models/land_use_classifier.joblib` | Path to the trained model artifact |
| `GALAXEYE_MAX_UPLOAD_SIZE_BYTES` | `5242880` (5 MB) | Max accepted upload size |

Example:
```bash
GALAXEYE_CONFIDENCE_THRESHOLD=0.75 uvicorn app.main:app --reload
```

## Design Decisions

See [`DESIGN.md`](DESIGN.md) for the full write-up: architecture, storage
model, confidence handling, trade-offs, assumptions, and open questions.

## Part 3

See [`PART3.md`](PART3.md) for the written answers on debugging, monitoring
an offline deployment, and design weaknesses.

## Limitations

- The model is a lightweight, hand-crafted-feature classifier, not a CNN —
  chosen deliberately for explainability and offline reproducibility (see
  DESIGN.md §6), at some accuracy cost on visually similar classes
  (Highway/River, specifically).
- `GET /results` supports the filters listed above; it does not paginate
  beyond a simple `limit`/`offset`, and there's no auth on any endpoint —
  both reasonable to add before any real deployment, out of scope here.
- The raw tile image is never stored, only its hash — see DESIGN.md §4 for
  the reasoning and what it would take to change that.
- SQLite is a single-writer database; it's the right choice for this scope
  but would need to become Postgres (or similar) under real concurrent
  write load.
- There's no automated drift detection running in this codebase — PART3.md
  Q2 describes how I'd approach it, but it isn't implemented here.

## Future Improvements

Realistic next steps if this moved toward production, without jumping
straight to heavy infrastructure:

- A scheduled "golden set" regression check (PART3.md Q2) as an actual
  script/cron job, not just a description.
- Optional persistence of raw tiles (flat files + an `image_path` column)
  if audit/reprocessing turns out to be needed.
- Swap SQLite for Postgres if concurrent write throughput becomes real.
- Move inference off the request thread (a simple background worker/queue)
  if tiles start arriving faster than they can be classified synchronously —
  not needed at the throughput this take-home implies.
- Revisit the model itself (more data, a small CNN) if accuracy on the
  weaker classes (Highway/River) turns out to matter operationally.
