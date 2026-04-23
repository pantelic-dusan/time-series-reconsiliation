"""Shared logging utilities for train/evaluate entry-point scripts.

Provides:
    * setup_logging(log_file) - configure the root logger to append to a single
      accumulating log file plus stderr.
    * timed(label)            - context manager that logs [START]/[DONE]/[FAIL]
      with the elapsed wall time of the wrapped block.
"""

from __future__ import annotations

import logging
import time
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
RUN_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


class _NoiseFilter(logging.Filter):
    """Suppress known low-signal third-party INFO chatter everywhere."""

    _NOISE_RULES = (
        ("cmdstanpy", "Chain [1] start processing"),
        ("cmdstanpy", "Chain [1] done processing"),
        ("prophet", "n_changepoints greater than number of observations."),
    )

    def filter(self, record: logging.LogRecord) -> bool:
        # Never hide warnings/errors.
        if record.levelno >= logging.WARNING:
            return True

        message = record.getMessage()
        for logger_prefix, snippet in self._NOISE_RULES:
            if record.name.startswith(logger_prefix) and snippet in message:
                return False

        return True


def setup_logging(
    log_file: str | Path,
    level: int = logging.INFO,
    console: bool = True,
    timestamped: bool = True,
    suppress_noisy_output: bool = True,
) -> logging.Logger:
    """Configure the root logger to append to `log_file` (accumulating).

    If `timestamped` is True (default), a ``_YYYYmmdd_HHMMSS`` suffix is
    inserted before the file extension so each run gets its own file while
    still living next to previous runs.

    Existing handlers are removed first so the function is idempotent and safe
    to call from multiple entry points in the same interpreter session.
    """
    log_path = Path(log_file)
    if timestamped:
        stamp = datetime.now().strftime(RUN_TIMESTAMP_FMT)
        log_path = log_path.with_name(f"{log_path.stem}_{stamp}{log_path.suffix}")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT, LOG_DATEFMT)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    # Third-party warning from GluonTS internals; not actionable in this repo.
    warnings.filterwarnings(
        "ignore",
        message=r"Using a non-tuple sequence for multidimensional indexing is deprecated.*",
        category=UserWarning,
        module=r"gluonts\.torch\.util",
    )

    noise_filter = _NoiseFilter() if suppress_noisy_output else None

    # Accumulating file handler - append mode, never truncated.
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    if noise_filter is not None:
        file_handler.addFilter(noise_filter)
    root_logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        if noise_filter is not None:
            stream_handler.addFilter(noise_filter)
        root_logger.addHandler(stream_handler)

    root_logger.info(f"=== Logging initialized -> {log_path.resolve()} ===")
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
        log.error(f"[FAIL] {label} after {elapsed:.1f}s - {exc!r}", exc_info=True)
        raise
    else:
        elapsed = time.perf_counter() - start
        log.info(f"[DONE] {label} in {elapsed:.1f}s")
