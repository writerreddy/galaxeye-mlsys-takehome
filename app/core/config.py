"""
Centralized application configuration.

Everything that might need to change between environments (or during a
live-coding session in the interview) lives here as a single Settings
object, read from environment variables with sensible defaults. This is
the first place to look if you need to change the confidence threshold,
the model path, or the database location.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root, resolved relative to this file so paths work regardless of
# the current working directory the app is started from.
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # --- App metadata ---
    app_name: str = "GalaxEye Tile Classification Service"
    model_version: str = "v1"

    # --- Storage ---
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'app.db'}"
    model_path: Path = BASE_DIR / "models" / "land_use_classifier.joblib"

    # --- Confidence policy ---
    # Predictions below this threshold are stored with status="uncertain"
    # instead of "accepted". See DESIGN.md section 5 for the reasoning;
    # this number is a placeholder default, not a validated optimum.
    confidence_threshold: float = 0.60

    # --- Upload validation ---
    max_upload_size_bytes: int = 5 * 1024 * 1024  # 5 MB
    allowed_content_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/jpg",
    )

    # --- Dataset locations (used by scripts/train.py and scripts/evaluate.py) ---
    dataset_dir: Path = BASE_DIR / "data" / "be-mlsys-assignment-dataset"
    train_dir: Path = dataset_dir / "candidate_tiles"
    eval_dir: Path = dataset_dir / "eval_set"
    eval_labels_csv: Path = dataset_dir / "eval_labels.csv"

    model_config = SettingsConfigDict(env_prefix="GALAXEYE_", protected_namespaces=())


settings = Settings()
