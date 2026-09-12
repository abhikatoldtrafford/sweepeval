"""Scorers (spec §11). Importing the package registers the shipped families."""

from __future__ import annotations

from sweepeval.scorers import deferred, guardrail, operational, security  # noqa: F401
from sweepeval.scorers.base import (
    SCORER_ENTRY_POINT_GROUP,
    ScoreContext,
    Scorer,
    ScorerRegistry,
    register,
    registry,
)

__all__ = [
    "SCORER_ENTRY_POINT_GROUP",
    "ScoreContext",
    "Scorer",
    "ScorerRegistry",
    "register",
    "registry",
]
