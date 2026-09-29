# REVIEW.md — Self-Review Against the Assignment

This is a line-by-line check of the submission against the actual assignment
requirements, plus anticipated interview questions with candidate-voice
answers.

## Requirement Coverage

| Requirement | Where it's implemented |
|---|---|
| Part 1: design note (architecture, components, tile lifecycle, storage/query strategy, confidence handling, trade-offs, assumptions, questions) | `DESIGN.md` |
| Part 2: one endpoint takes a tile, classifies it locally, stores the result | `app/api/routes.py::classify_tile` → `app/services/classification_service.py::classify_and_store` |
| README explaining how to run it | `README.md` |
| Part 3: written answers | `PART3.md` |
| Offline, no hosted model APIs | `app/ml/classifier.py` loads a local `.joblib` file only; no HTTP client anywhere in `app/` |
| Locally runnable model on CPU | `scripts/train.py` trains a `sklearn.ensemble.RandomForestClassifier`; inference is ~10-20ms on CPU |
| Not grading on accuracy, but honest metrics | `scripts/evaluate.py` prints real measured numbers (85.2% eval accuracy) — see README's "Running Evaluation" section, numbers not fabricated |
| Confidence handling | `app/services/classification_service.py::classify_and_store` (threshold check), `GALAXEYE_CONFIDENCE_THRESHOLD` in `app/core/config.py` |
| Result storage | `app/db/models.py::ClassificationResult`, SQLite via `app/db/database.py` |
| Analyst querying (optional, implemented) | `GET /results` with filters in `app/api/routes.py::list_results` |
| Validation (file exists, valid image, format, size) | `app/ml/preprocessing.py::load_image_from_bytes`, `app/services/classification_service.py::_validate_upload` |
| Structured error responses, correct status codes | `app/api/routes.py::classify_tile` (400/413/500), `app/main.py` global exception handler |
| Logging, no swallowed exceptions | `app/core/logging.py`, `logger.exception(...)` calls in `app/api/routes.py` and `app/main.py` |
| Tests: health, classification, validation, database, integration | `tests/test_health.py`, `tests/test_classification.py`, `tests/test_validation.py`, `tests/test_database.py` (22 tests total, all passing) |
| Reproducible training + evaluation scripts | `scripts/train.py`, `scripts/evaluate.py` |
| No training at API startup | `app/main.py` only calls `classifier.load()`, never a training function |

## Potential Interview Questions

**Why FastAPI?**
Async-capable, built-in request validation via Pydantic, and it generates
interactive docs for free, which is genuinely useful when someone else
(you, right now) needs to poke at the API. It's also the framework I'm
fastest and most confident in, which mattered given the time box.

**Why SQLite?**
Single file, zero setup, trivially copyable onto isolated hardware — which
matches the "offline, isolated hardware" scenario in the assignment
directly. It's the wrong choice the moment there's real concurrent write
load, and I say so in DESIGN.md — but nothing here suggests that load yet.

**Why this ML model instead of a pretrained CNN?**
Two reasons. First, the assignment is explicit that accuracy isn't the
primary grading criterion — engineering reasoning is. A hand-crafted-feature
model is something I can point at, line by line, and explain exactly why it
produces the number it produces; a pretrained CNN's internals are much
harder to defend live in an interview. Second, it trains in under a second
on CPU with no external weights to manage, which fits the offline
constraint cleanly. If accuracy on the harder classes (Highway/River)
became a real requirement, a small CNN trained on this same data would be
the natural next step — I'd just want to be sure that trade against
explainability was worth it for the actual use case first.

**Why not a pretrained CNN?**
Same answer as above — it's not that I couldn't, it's that classical
features gave a defensible, fast, CPU-only baseline that matches what the
assignment says it's grading. I'd genuinely enjoy discussing the CNN
version live if that's useful context.

**How does confidence work?**
The model's `predict_proba` gives a probability for each class; I take the
top one as `confidence`. If it's at or above a configurable threshold
(0.60 by default), the result is `status: accepted`; otherwise
`status: uncertain`. Both the prediction and the confidence are always
returned — the caller decides what to do with an uncertain one.

**How would you calibrate confidence properly?**
Right now `confidence` is just the raw max class probability from the
forest's vote distribution — it's a reasonable ranking signal (as the
eval numbers show) but it isn't a formally calibrated probability. To
calibrate it properly I'd use `sklearn.calibration.CalibratedClassifierCV`
(Platt scaling or isotonic regression) fit on a held-out calibration split,
so that "70% confidence" actually means "correct about 70% of the time" in
a measurable sense, not just "higher means more likely correct."

**Why store image hash?**
It's a cheap way to answer "did this exact input produce this result?"
without storing the image itself — useful for debugging (comparing two
"identical" tiles) and dedup, at effectively zero storage cost.

**Why store model version?**
So that if the model is ever retrained or swapped, old and new predictions
in the same table remain distinguishable. Without it, a silent model change
would be invisible in the data.

**What happens if the model file disappears?**
The app doesn't crash at startup — `app/main.py`'s lifespan handler catches
the load failure and logs it. `/health` reports `model_loaded: false`, and
`POST /tiles/classify` returns a clean `500` rather than a stack trace or a
hard crash. I'd rather have a running-but-degraded service that's
observably broken than a process that won't start at all.

**How would you scale this?**
Depends what "scale" means. More requests per second at similar latency:
add more API worker processes behind a load balancer (SQLite becomes the
bottleneck here, so that's also when Postgres comes in). Much larger
result volume: partition/archive old rows, or move to Postgres for better
concurrent write handling. Either way, I'd want real throughput numbers
before picking a specific fix — see the questions in DESIGN.md §10.

**How would you handle 1000 tiles/sec?**
That's well past what a single synchronous FastAPI process handles cleanly.
I'd decouple ingestion from inference: an endpoint that quickly accepts and
queues the tile, a pool of worker processes doing the actual classification,
and a batched/bulk write path into the database instead of one commit per
tile. That's a meaningfully different architecture than this take-home's
scope, which is explicitly a thin slice — but it's a natural evolution, not
a rewrite of the model or storage schema.

**How would you process tiles asynchronously?**
A lightweight queue (even a simple on-disk work queue, given the offline
constraint — no Kafka/Redis without a concrete need) between ingestion and
inference, with the API returning immediately with a "processing" status
and a way to poll for the result by `tile_id`. The `ClassificationResult`
table's shape barely changes; I'd add a `pending` status alongside
`accepted`/`uncertain`.

**What happens if the database is unavailable?**
Right now, `db.commit()` would raise, and the outer handler in
`classify_tile` catches unexpected exceptions and returns a `500` without
leaking internals — see the `except Exception` block in `app/api/routes.py`.
It's logged with the real error server-side. I don't currently retry;
for something running unattended, a bounded retry with backoff before
failing would be a reasonable addition.

**How do you detect model drift offline?**
Without new ground truth, I can't measure accuracy drift directly, so I'd
watch proxies: the distribution of predicted classes and confidence scores
over time (a sudden shift suggests either the world changed or the model's
behavior changed), plus a periodic "golden set" of known tiles re-run
through the same model to catch corruption or accidental swaps. Full answer
in PART3.md Q2.

**How would you update the model?**
Retrain with `scripts/train.py` (or a better model altogether), validate it
against `eval_set/` with `scripts/evaluate.py`, and if the numbers hold up,
replace `models/land_use_classifier.joblib` and bump `model_version` in
config. Because `model_version` is stamped per stored row, old and new
predictions stay distinguishable in the data after the swap.

**How would you roll back a model?**
Keep the previous `.joblib` artifact around (e.g., versioned filenames) and
swap the file back, reverting `model_version`. Because nothing about
inference is stateful beyond that one file, rollback is just "point at the
old file again" — no data migration needed.

**How do you know inference results are reproducible?**
`scripts/train.py` fixes a random seed, and `extract_features` is a pure,
deterministic function (no randomness) — the same image always produces the
same feature vector and, given the same model file, the same prediction. I
wrote `test_extract_features_is_deterministic` in `tests/test_validation.py`
specifically to pin that down.

**What happens when the model is uncertain?**
The prediction is still returned, but with `status: "uncertain"` — nothing
is hidden or discarded. It's on the caller to decide what "uncertain" means
for their workflow (human review, exclude from aggregates, flag for
re-imaging, etc.) — that policy decision isn't mine to make without knowing
the downstream use.

**How do you test ML code?**
Two layers, deliberately kept separate. Unit tests (most of
`tests/test_classification.py`, all of `tests/test_database.py`) use a
`FakeClassifier` with fixed, known outputs, so they test the API/service/
storage logic without depending on real model accuracy at all. Separately,
one true integration test
(`test_classification.py::test_full_pipeline_with_real_model`) loads the
actual trained artifact and runs a real image through the whole stack, to
prove the real pieces actually fit together. `scripts/evaluate.py` is the
model-quality check, run manually/on a schedule, not as part of the fast
unit test suite.

**What's the biggest weakness?**
Answered in full in PART3.md Q4: the model has no way to notice its own
input distribution has shifted, and there's no ground truth after
deployment to catch that automatically. That's the thing I'd prioritize
fixing first if this went to production, ahead of chasing a better model.

**How would you make this production-ready?**
In rough priority order: get the drift/golden-set monitoring from PART3.md
Q2 actually running on a schedule; add auth to the API; decide on the
image-retention question in DESIGN.md §10 and implement whichever way that
goes; move to Postgres if concurrent write load turns out to be real; and
only then revisit the model itself if the weaker classes (Highway/River)
turn out to matter for the actual use case.

## Live-Coding Preparation

Five plausible small live changes, and exactly where each one goes:

1. **Change the confidence threshold.**
   `app/core/config.py` → `confidence_threshold` field (or set
   `GALAXEYE_CONFIDENCE_THRESHOLD` as an env var, no code change needed at
   all). The comparison itself lives in
   `app/services/classification_service.py::classify_and_store`.

2. **Add a new filter to `GET /results`** (e.g. filter by `filename`).
   Add a query parameter in `app/api/routes.py::list_results`, and one more
   `stmt = stmt.where(...)` line — the pattern is already established for
   `predicted_class`, `status`, `min_confidence`, and the date range.

3. **Add a new field to the classification response** (e.g.
   `feature_vector_length`).
   Add the field to `ClassificationResponse` in
   `app/schemas/classification.py`, then populate it in the return
   statement of `classify_tile` in `app/api/routes.py`. If it needs to be
   persisted too, add a column to `ClassificationResult` in
   `app/db/models.py` and set it in `classify_and_store`.

4. **Add validation for another image format** (e.g. WEBP).
   Add the MIME type to `allowed_content_types` in `app/core/config.py`.
   `load_image_from_bytes` in `app/ml/preprocessing.py` already delegates
   decoding to Pillow, which supports WEBP natively — no change needed
   there.

5. **Add an endpoint to retrieve a result by tile ID.**
   Already implemented (`GET /results/{tile_id}` in `app/api/routes.py`) —
   a good one to point to directly as a worked example of "this is what
   adding an endpoint looks like here" before making a different live
   change.
