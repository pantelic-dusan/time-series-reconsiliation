"""Shared logging utilities for train/evaluate entry-point scripts.

Provides:
    * setup_logging(log_file) — configure the root logger to append to a single
      accumulating log file plus stderr.
    * timed(label)            — context manager that logs [START]/[DONE]/[FAIL]
      with the elapsed wall time of the wrapped block.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_file: str | Path,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure the root logger to append to `log_file` (accumulating).

    Existing handlers are removed first so the function is idempotent and safe
    to call from multiple entry points in the same interpreter session.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, LOG_DATEFMT)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Accumulating file handler — append mode, never truncated.
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        root_logger.addHandler(stream_handler)

    root_logger.info(f"=== Logging initialized → {log_path.resolve()} ===")
    return root_logger


@contextmanager
def timed(label: str, logger: Optional[logging.Logger] = None) -> Iterator[None]:
    """Log [START]/[DONE]/[FAIL] around a block and report its elapsed time.

    Re-raises any exception after logging it (with traceback) at ERROR level.
    """
    log = logger or logging.getLogger(__name__)
    start = time.perf_counter()
    log.info(f"[START] {label}")
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        log.error(f"[FAIL] {label} after {elapsed:.1f}s — {exc!r}", exc_info=True)
        raise
    else:
        elapsed = time.perf_counter() - start
        log.info(f"[DONE] {label} in {elapsed:.1f}s")

