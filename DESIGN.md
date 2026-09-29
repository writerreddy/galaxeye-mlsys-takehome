# GalaxEye Offline Satellite Tile Classification — Design Note

## 1. Goal

Ingest a satellite image tile, classify its land-use type (forest, water,
residential, ...) using a model that runs entirely on local hardware, store
the result durably, and let an analyst query stored results later — all with
no internet access at runtime. This document covers the design for the full
system; the accompanying code implements a thin, working slice of it
(Part 2), not the whole thing.

## 2. Proposed Architecture

```
        Tile (image bytes)
              |
              v
        FastAPI endpoint (POST /tiles/classify)
              |
              v
        Input validation (content-type, size, decodability)
              |
              v
        Preprocessing (resize, feature extraction)
              |
              v
        Local ML model (in-process, CPU, pre-trained artifact on disk)
              |
              +--> predicted_class
              +--> confidence
              +--> model_version
              |
              v
        Confidence policy (accepted / uncertain)
              |
              v
        Persistence layer (SQLAlchemy ORM)
              |
              v
        SQLite (single file on local disk)
              |
              v
        Analyst queries (GET /results?predicted_class=...&min_confidence=...)
```

Everything runs in a single Python process. There is no message queue, no
separate inference service, and no distributed storage — see section 6 for
why, and section 7 for when that would change.

## 3. Tile Lifecycle

1. A client sends a tile as `multipart/form-data` to `POST /tiles/classify`.
2. The API layer reads the raw bytes and validates: is it a supported
   content type, is it under the size limit, does it actually decode as an
   image? Any failure returns a 4xx with a clear message; nothing is
   persisted.
3. The image is resized to a fixed size and turned into a feature vector
   (color histograms + brightness/contrast + a coarse texture signal — see
   `app/ml/preprocessing.py`).
4. The pre-loaded classifier (loaded once at process startup, never
   retrained per-request) produces a probability distribution over classes.
   The top class and its probability become `predicted_class` and
   `confidence`.
5. The confidence policy compares `confidence` to a configurable threshold
   and assigns `status`: `accepted` or `uncertain`.
6. A row is written to `classification_results` with the prediction, a
   SHA-256 hash of the input bytes, the model version, and timing
   information, then committed.
7. The API returns `tile_id`, `predicted_class`, `confidence`, `status`,
   `model_version`, and `processing_time_ms` to the caller.
8. Later, an analyst can call `GET /results` with filters, or
   `GET /results/{tile_id}` for a single record.

## 4. Storage Model

**What's stored, per tile:** a generated `tile_id` (UUID, stable even if
filenames collide), the original `filename`, a SHA-256 `image_hash` of the
raw bytes, `predicted_class`, `confidence`, `status`, `model_version`,
`processing_time_ms`, and `processed_at`.

**What's *not* stored:** the raw image bytes themselves. For a take-home
slice, storing images in SQLite (as BLOBs) or on disk adds real cost
(storage growth, backup size, no clear retention policy) for a benefit —
audit/reprocessing — that the assignment doesn't ask for and that I can't
size without knowing GalaxEye's actual retention/audit requirements (see
section 10). The `image_hash` is the compromise: it's enough to prove
"this exact input produced this exact output" and to detect duplicate
uploads, without committing to a storage strategy for the images themselves.
If audit or reprocessing turns out to matter, the natural extension is to
store tiles in local flat files (or object storage, if network access is
ever available) and add an `image_path` column — additive, not a rewrite.

**Indexes:** `predicted_class`, `status`, `confidence`, `processed_at`, and
`image_hash` are all indexed individually, plus a composite index on
`(predicted_class, confidence)` for the "class X above confidence Y" query
pattern I expect analysts to use most. This is a deliberately generous set
of indexes for what will likely stay a small table in this take-home
context; on a much larger table I'd measure actual query patterns before
keeping all of them, since indexes aren't free on writes.

**Model version** is stored per-row (not just globally) so that if the
model is retrained or replaced, old and new predictions in the same table
stay distinguishable — important for both debugging (Part 3, Q3) and drift
monitoring (Part 3, Q2).

## 5. Confidence / Uncertainty

The classifier returns a probability for its top class. I compare that
probability to `CONFIDENCE_THRESHOLD` (default `0.60`, configurable via
env var) and mark the result `accepted` or `uncertain` accordingly.

I want to be explicit that **0.60 is a reasonable-looking default, not a
scientifically derived optimum.** The right threshold depends on the
operational cost of a false positive vs. a false negative for whatever
downstream decision consumes these labels — something only GalaxEye can
tell me. What I *can* say from the numbers I actually measured (see
`scripts/evaluate.py` output, and PART3.md Q1) is that on the held-out
`eval_set`, predictions at or above 0.60 confidence are right about 97% of
the time, while predictions below it are right only about half the time.
That's a genuinely useful split in practice, even though the threshold
itself was picked by eyeballing, not by optimizing a cost function I don't
have.

Exposing `confidence` and `status` separately (rather than just silently
dropping low-confidence predictions) is a deliberate choice: it lets
downstream consumers decide their own policy — route "uncertain" tiles to
a human, log them separately, exclude them from aggregate stats — without
GalaxEye needing to re-run inference to get that information.

## 6. Trade-offs

| Decision | Option | Advantage | Disadvantage | Chosen | Why |
|---|---|---|---|---|---|
| Database | SQLite vs PostgreSQL | SQLite: zero setup, single file, trivially backed up/copied to isolated hardware | SQLite: limited write concurrency, no network access for remote analysts | **SQLite** | Take-home scope, single-process, offline deployment; no evidence yet of concurrent-write load that would need Postgres |
| API style | Synchronous REST vs async queue | Sync: simple, immediate feedback, easy to test and demo | Sync: caller blocks for the full inference time; no natural back-pressure under burst load | **Synchronous REST** | The assignment describes a single endpoint returning a result; inference here is ~10-20ms, so blocking is a non-issue at this scale |
| Model | Local lightweight model vs remote/cloud model | Local: works with zero network dependency, matches the "isolated hardware" scenario exactly | Local: capped by CPU inference and whatever data is available for training | **Local model, CPU inference** | Ground rules explicitly forbid hosted/cloud model APIs at runtime |
| Model complexity | Hand-crafted features + classical ML vs CNN | Classical ML: trains in under a second, no GPU, every feature is literally a few lines of NumPy I can point to, trivial to modify live | CNN: could plausibly reach higher accuracy with more data/tuning | **Hand-crafted features + RandomForest** | Assignment explicitly says accuracy isn't the primary grading criterion; explainability and live-modifiability matter more here. See section 7 for how I'd revisit this. |
| Image storage | Store raw image vs metadata only | Store image: enables audit/reprocessing/debugging by re-running old inputs | Store image: storage growth, no clear retention need stated | **Metadata + hash only** | Not asked for; hash preserves traceability at near-zero cost. Easy to add later without breaking existing rows. |
| Prediction reporting | Hard prediction only vs prediction + uncertain state | Hard only: simpler consumer contract | Hard only: silently reports low-confidence guesses as if certain | **Prediction + status field** | Assignment explicitly asks how I'd handle predictions the model isn't confident about |

## 7. Offline Constraints

- No code path makes a network call at runtime. The one place a "network"
  concept could sneak in — model weights — is handled by training once
  and saving a local artifact (`models/land_use_classifier.joblib`) that
  the app loads from disk at startup; nothing is downloaded.
- Dependencies are pinned to exact versions in `requirements.txt` so a
  fresh environment reproduces the same behavior.
- The training script (`scripts/train.py`) is deterministic (fixed random
  seed) so the artifact can be regenerated and gives the same result.
- `model_version` is stored on every row specifically so that, on
  isolated hardware where a human has to manually copy a new model file
  over, old and new predictions remain distinguishable after the swap.

## 8. Reliability / Observability

For a fully offline deployment nobody is watching live, the things worth
having are the things that let you retroactively answer "is this still
working?" without needing live access or ground truth:

- **Structured logs** (`app/core/logging.py`) — every classification logs
  `tile_id`, predicted class, confidence, status, and latency, one line
  per event, to stdout. On a real deployment this would be redirected to
  a rotated file.
- **Persisted results table** doubles as a metrics source: prediction
  volume over time, class distribution, confidence distribution, and
  `uncertain` rate can all be computed with a `GROUP BY` against
  `classification_results` — no separate metrics pipeline needed at this
  scale.
- **Processing latency** is stored per-row, so a creeping latency problem
  (disk filling up, thermal throttling, etc.) is visible in the data
  itself.
- **Model version** stored per-row, so a silent, accidental model swap
  shows up immediately in the data instead of needing to be remembered.
- Longer-term, offline drift detection would rely on comparing the
  *distribution* of predicted classes and confidence scores over time
  against a known-good baseline (since true labels aren't available after
  deployment) — see PART3.md Q2 for the full answer.

## 9. Assumptions

- "One endpoint that takes a tile" means a single-image, synchronous
  classify endpoint, not a batch/bulk upload endpoint.
- The supplied `candidate_tiles/` directory is fair to train on, and
  `eval_set/` + `eval_labels.csv` is a held-out set meant for evaluating
  the trained model (its own README.txt says as much) — so training uses
  only `candidate_tiles/`, and `eval_set/` is used only by
  `scripts/evaluate.py`, never for training.
- "Analyst querying" for the take-home means a simple filterable HTTP
  endpoint (`GET /results`) against the same SQLite database, not a
  separate analytics store or BI tool integration.
- A single SQLite file, colocated with the API process, is an acceptable
  storage choice for a take-home / small-deployment scope; I have not
  assumed anything about GalaxEye's actual production scale.
- Confidence thresholding is the primary uncertainty-handling mechanism
  expected here, rather than e.g. calibrated probability intervals or an
  ensemble/abstention model — the simpler mechanism that's still genuinely
  useful, appropriate for the scope of this exercise.
- I assumed PNG/JPEG as the supported upload formats, matching the
  supplied dataset; other formats are rejected with a 400 rather than
  silently coerced.

## 10. Questions for GalaxEye

- What throughput is expected in production — tiles per second/minute, and
  is it bursty (e.g. after a satellite pass) or steady?
- What hardware will this actually run on (CPU-only? GPU available? RAM/
  disk constraints on the isolated machines)?
- What inference latency is acceptable per tile?
- Are these classifications advisory (a human reviews them) or do they
  feed directly into an automated downstream decision? This changes how
  conservative the confidence threshold should be.
- Is there a real cost asymmetry between error types (e.g. missing a
  genuine `Highway` is worse than mislabeling `AnnualCrop` as `Forest`)
  that should shape the threshold or even per-class thresholds?
- How are model updates physically distributed to isolated/offline
  hardware, and is there a process for validating a new model artifact
  before it's put into production there?
- Is the original tile image needed later for audit, reprocessing, or
  retraining, or is the classification result itself the durable
  artifact?
- What retention period applies to stored results (and to tiles, if they
  end up being stored)?
- Will any ground-truth labels ever become available after deployment
  (e.g. from later human review), or should monitoring assume label-free
  operation indefinitely?
- What would GalaxEye consider an unacceptable model failure rate for this
  use case — is 30% wrong ever acceptable if it's the *right* 30%, or is
  there a hard floor?
