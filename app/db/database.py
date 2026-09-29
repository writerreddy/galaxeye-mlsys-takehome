"""
SQLAlchemy engine / session wiring.

Kept intentionally small: one engine, one sessionmaker, one FastAPI
dependency (`get_db`) that yields a session per-request and always
closes it. This is the standard FastAPI + SQLAlchemy pattern and is easy
to swap to Postgres later by changing `settings.database_url` alone —
nothing else in the app needs to know which database is behind it.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

# `check_same_thread` is only needed for SQLite: FastAPI can hand a
# request to a different thread than the one that created the
# connection, and SQLite's default driver forbids that. It's a no-op
# for other engines.
connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """Create tables if they don't exist yet. Safe to call on every startup."""
    # Import models here so they're registered on Base.metadata before
    # create_all runs, without creating an import cycle at module load time.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
