# data/

- `be-mlsys-assignment-dataset/` — the dataset supplied with the assignment,
  unmodified (see its own `README.txt` inside for GalaxEye's original
  description). Contains:
  - `candidate_tiles/<ClassName>/*.png` — 150 labeled tiles per class (7
    classes, 1050 total), used only for training (`scripts/train.py`).
  - `eval_set/*.png` + `eval_labels.csv` — 210 labeled tiles held out for
    evaluation only (`scripts/evaluate.py`); never used for training.
- `app.db` — the runtime SQLite database. Not checked in (see
  `.gitignore`); created automatically the first time the API starts.
