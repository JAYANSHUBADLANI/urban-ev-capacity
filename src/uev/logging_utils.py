"""Logging configured once, writing repository relative paths only."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

_CONFIGURED = False


def get_logger(name: str = "uev") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("uev")
        root.setLevel(logging.INFO)
        root.handlers = [handler]
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("uev") else f"uev.{name}")


@contextmanager
def timed(label: str, logger: logging.Logger | None = None):
    """Time a block and report the elapsed seconds."""
    log = logger or get_logger()
    start = time.perf_counter()
    log.info("start %s", label)
    try:
        yield
    finally:
        log.info("done %s in %.1fs", label, time.perf_counter() - start)
