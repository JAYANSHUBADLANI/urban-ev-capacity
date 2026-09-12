"""Deterministic seeding.

A single base seed is combined with a label so that independent components draw
independent but reproducible streams.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

from .config import SEED


def derive_seed(label: str, base: int = SEED) -> int:
    """Derive a stable 32 bit seed from a label and the base seed."""
    digest = hashlib.sha256(f"{base}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def rng(label: str, base: int = SEED) -> np.random.Generator:
    """Return a numpy generator seeded from the label."""
    return np.random.default_rng(derive_seed(label, base))


def seed_everything(base: int = SEED) -> None:
    """Seed the global interpreter level generators."""
    random.seed(base)
    np.random.seed(base % (2 ** 32))
