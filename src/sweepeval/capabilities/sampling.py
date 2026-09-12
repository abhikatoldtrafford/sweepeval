"""The sampling-effect test (spec §9.1, D7).

The single most important capability check. If temperature does nothing,
sweeping it is theatre; if the test wrongly says it does nothing, the whole
experiment is silently truncated.

Three properties this implementation is built around, each fixing a specific
defect in an earlier revision of the spec:

* **A full decision table.** Rev 1 defined two cells — "high≥3 and low=1" and
  "both=1" — and left seven others undefined. The most common real case had no
  cell at all: MoE routing, batching and GPU nondeterminism make ``temp=0``
  non-deterministic on most hosted endpoints.
* **Normalised, per-prompt distinctness.** Pooled across prompts, two different
  prompts trivially give two distinct outputs and a single deviation declares
  ``EFFECTIVE``. Raw strings make every response distinct on any target that
  echoes a request id.
* **``INERT`` requires positive evidence.** Not-significant is
  ``INCONCLUSIVE``, never ``INERT``. Absence of evidence must not remove an
  axis from the experiment, so ``INERT`` is reached only by a TOST equivalence
  result against a declared margin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sweepeval.capabilities.normalise import distinct_count, token_jaccard
from sweepeval.stats.equivalence import (
    dispersion as _dispersion,
)
from sweepeval.stats.equivalence import (
    dispersion_difference_test,
)
from sweepeval.stats.equivalence import (
    tost_equivalent as _tost,
)

__all__ = [
    "TIER1_RUNS",
    "TIER2_RUNS",
    "TOST_MARGIN",
    "SamplingVerdict",
    "Verdict",
    "decide_tier1",
    "dispersion",
    "tost_equivalent",
]

TIER1_RUNS = 3
TIER2_RUNS = 8

TOST_MARGIN = 0.05
"""Equivalence margin in normalised-Jaccard dispersion (§9.1).

Fixed by the spec and surfaced in the manifest rather than exposed as a flag,
so a future change to it is visible in stored runs.
"""


class Verdict(str, Enum):
    EFFECTIVE = "EFFECTIVE"
    INERT = "INERT"
    INCONCLUSIVE = "INCONCLUSIVE"
    ESCALATE = "ESCALATE"
    """Tier 1 only: not a final verdict."""


@dataclass
class SamplingVerdict:
    parameter: str
    verdict: Verdict
    tier: int
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    @property
    def swept(self) -> bool:
        """§9.1: EFFECTIVE and INCONCLUSIVE are swept; INERT is not.

        Including an inert axis costs money. Excluding an effective one
        silently destroys the experiment. The asymmetry is deliberate.
        """
        return self.verdict in (Verdict.EFFECTIVE, Verdict.INCONCLUSIVE)


def decide_tier1(
    low_outputs: Sequence[Sequence[str]],
    high_outputs: Sequence[Sequence[str]],
) -> SamplingVerdict:
    """Apply the §9.1 decision table, per prompt, on normalised text.

    Each argument is a sequence of per-prompt run lists: ``[[r1, r2, r3], ...]``.
    """
    if len(low_outputs) != len(high_outputs) or not low_outputs:
        raise ValueError("need matching, non-empty per-prompt run lists")

    per_prompt: list[tuple[int, int]] = [
        (distinct_count(list(low)), distinct_count(list(high)))
        for low, high in zip(low_outputs, high_outputs, strict=True)
    ]

    verdicts = [_cell(low, high) for low, high in per_prompt]
    evidence = {
        "per_prompt_distinct": [
            {"low": low, "high": high, "cell": v.value}
            for (low, high), v in zip(per_prompt, verdicts, strict=True)
        ],
        "runs_per_setting": len(low_outputs[0]),
    }

    # Any prompt showing a clear effect settles it: an axis that moves output
    # on one open-ended prompt is not inert.
    if Verdict.EFFECTIVE in verdicts:
        return SamplingVerdict(
            parameter="",
            verdict=Verdict.EFFECTIVE,
            tier=1,
            evidence=evidence,
            reason="at least one prompt showed a clear increase in distinct outputs",
        )

    return SamplingVerdict(
        parameter="",
        verdict=Verdict.ESCALATE,
        tier=1,
        evidence=evidence,
        reason="tier 1 could not separate; escalating to the dispersion test",
    )


def _cell(distinct_low: int, distinct_high: int) -> Verdict:
    """One row of §9.1's table.

    ``distinct_low == 3`` is the row rev 1 had no cell for: the target is
    already non-deterministic at temp=0, so counting distinct outputs cannot
    separate the settings and only the dispersion test can.
    """
    if distinct_low >= 3:
        return Verdict.ESCALATE
    if distinct_low == 1 and distinct_high >= 3:
        return Verdict.EFFECTIVE
    if distinct_low == 2 and distinct_high >= 3:
        return Verdict.EFFECTIVE
    return Verdict.ESCALATE


def dispersion(outputs: Sequence[str]) -> float:
    """Mean pairwise dissimilarity over the runs, on normalised text.

    Thin wrapper: the arithmetic lives in ``stats/`` because the layering
    contract keeps every p-value and interval there, and this module owns the
    decision table rather than the statistics.
    """
    return _dispersion(outputs, token_jaccard)


def tost_equivalent(
    low_outputs: Sequence[str],
    high_outputs: Sequence[str],
    *,
    margin: float = TOST_MARGIN,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[bool, dict[str, Any]]:
    """Two one-sided tests for equivalence of dispersion.

    Equivalence — not merely a failure to detect a difference — is what
    licenses ``INERT``.
    """
    equivalent, evidence = _tost(
        low_outputs, high_outputs, token_jaccard, margin=margin, alpha=alpha, seed=seed
    )
    evidence["dispersion_low"] = round(dispersion(low_outputs), 4)
    evidence["dispersion_high"] = round(dispersion(high_outputs), 4)
    evidence["observed_difference"] = round(
        dispersion(high_outputs) - dispersion(low_outputs), 4
    )
    return equivalent, evidence


def decide_tier2(
    parameter: str,
    low_outputs: Sequence[str],
    high_outputs: Sequence[str],
    *,
    margin: float = TOST_MARGIN,
    alpha: float = 0.05,
    seed: int = 0,
) -> SamplingVerdict:
    """Final verdict from the dispersion test."""
    p_superior, evidence = dispersion_difference_test(
        low_outputs, high_outputs, token_jaccard, seed=seed
    )
    equivalent, tost_evidence = tost_equivalent(
        low_outputs, high_outputs, margin=margin, alpha=alpha, seed=seed
    )
    evidence = {"p_high_disperses_more": round(p_superior, 4), **evidence, **tost_evidence}

    if p_superior < alpha:
        return SamplingVerdict(
            parameter=parameter,
            verdict=Verdict.EFFECTIVE,
            tier=2,
            evidence=evidence,
            reason="dispersion is significantly higher at the high setting",
        )

    if equivalent:
        return SamplingVerdict(
            parameter=parameter,
            verdict=Verdict.INERT,
            tier=2,
            evidence=evidence,
            reason=(
                f"dispersion is statistically equivalent within ±{margin} "
                "(TOST); the parameter has no measurable effect"
            ),
        )

    return SamplingVerdict(
        parameter=parameter,
        verdict=Verdict.INCONCLUSIVE,
        tier=2,
        evidence=evidence,
        reason=(
            "neither a significant difference nor equivalence; the axis is "
            "swept and flagged, because excluding an effective axis silently "
            "truncates the experiment while including an inert one only costs "
            "money"
        ),
    )


__all__ += ["decide_tier2"]
