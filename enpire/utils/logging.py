"""Logging helpers: a console logger and a JSONL run logger.

The :class:`RunLogger` writes one JSON object per research event to a ``.jsonl``
file, giving a complete, auditable record of the idea tree (every hypothesis,
its parent, the agent that proposed it, and the rollout result) -- the
simulation analogue of ENPIRE's Git-based experiment history.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any


def get_console_logger(name: str = "enpire", level: int = logging.INFO) -> logging.Logger:
    """Return a configured console logger (idempotent across calls)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


class RunLogger:
    """Append-only JSONL logger for a single autoresearch run.

    Args:
        path: Destination ``.jsonl`` file. Parent directories are created.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate any previous content so each run starts clean.
        self._fh = self.path.open("w", encoding="utf-8")

    def log(self, event: dict[str, Any]) -> None:
        """Write one event as a JSON line and flush."""
        self._fh.write(json.dumps(event) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
