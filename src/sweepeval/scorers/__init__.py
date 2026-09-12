"""Scorers (spec §11). Importing the package registers the shipped families."""

from __future__ import annotations

from sweepeval.scorers import (  # noqa: F401
    context,
    deferred,
    determinism,
    guardrail,
    operational,
    security,
)
from sweepeval.scorers.base import (
    SCORER_ENTRY_POINT_GROUP,
    CrossRunScorer,
    RunEvidence,
    ScoreContext,
    Scorer,
    ScorerRegistry,
    register,
    registry,
)

__all__ = [
    "SCORER_ENTRY_POINT_GROUP",
    "CrossRunScorer",
    "RunEvidence",
    "ScoreContext",
    "Scorer",
    "ScorerRegistry",
    "register",
    "registry",
]
