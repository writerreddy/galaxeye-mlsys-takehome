"""
Minimal structured-ish logging setup.

We deliberately don't pull in a logging framework. A single configured
stdlib logger, with one line per event including key fields, is enough
for a take-home slice and is exactly what you'd grep through when
debugging offline (see PART3.md, question 3).
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        # Avoid adding duplicate handlers if called more than once
        # (e.g. once from app import, once from a script import).
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
