"""Logging setup: a ``rich`` console handler; level from the ``FOODLENS_LOG`` env variable."""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

LOG_ENV_VAR = "FOODLENS_LOG"
DEFAULT_LEVEL = "INFO"
_ROOT_NAME = "foodlens"


def _configure_root() -> logging.Logger:
    root = logging.getLogger(_ROOT_NAME)
    level_name = os.environ.get(LOG_ENV_VAR, DEFAULT_LEVEL).upper()
    level = logging.getLevelName(level_name)
    if not isinstance(level, int):
        raise ValueError(f"{LOG_ENV_VAR}={level_name!r} is not a valid logging level")
    root.setLevel(level)
    if not any(isinstance(h, RichHandler) for h in root.handlers):
        handler = RichHandler(rich_tracebacks=True, show_path=False)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root.addHandler(handler)
        root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``foodlens`` namespace (e.g. ``foodlens.qc.frame_qc``)."""
    root = _configure_root()
    if name == _ROOT_NAME:
        return root
    if not name.startswith(_ROOT_NAME + "."):
        name = f"{_ROOT_NAME}.{name}"
    return logging.getLogger(name)
