"""Metric atoms (spec §6.4, §13.2, §13.4). I3 load-bearing.

I3: "Every metric carries an interval, or is explicitly marked
``NO_VALID_INTERVAL``. Never a bare point value."

The enforcement is structural rather than conventional: ``MetricValue`` is the
only type an aggregate field accepts, and it cannot be constructed without
either bounds or an explicit, flagged refusal to give them. Code that wants a
plain number has to write ``.point``, which makes the omission visible in
review instead of invisible in a report.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = ["CIMethod", "Estimand", "Flag", "MetricSpec", "MetricValue"]


class Flag(str, Enum):
    """Qualifiers attached to a measured value."""

    LOW_N = "LOW_N"
    """Fewer clusters than the bootstrap floor; the interval is a t-interval."""

    INDICATIVE = "INDICATIVE"
    """The interval is valid but wide, so few pairs will separate.

    Set on quick-profile results (§10.2) and on per-cell breakdowns (§13.7).
    It does **not** mean "no valid interval" — that is
    :attr:`NO_VALID_INTERVAL`. Conflating the two would let a report imply a
    number is unusable when it is merely imprecise (§6.4, rev 2.1a).
    """

    CACHE_SUSPECTED = "CACHE_SUSPECTED"
    """Repeated body hashes with implausible TTFT; determinism and latency
    derived from these runs are not trustworthy (§12.7)."""

    LOW_COVERAGE = "LOW_COVERAGE"
    """More than 30% of the family's trials were unscorable (§11.8)."""

    NO_VALID_INTERVAL = "NO_VALID_INTERVAL"
    """Too few clusters for any honest interval (§13.4)."""

    TTFT_REASONING_ADJUSTED = "TTFT_REASONING_ADJUSTED"
    """First token was reasoning rather than answer; TTFT was adjusted (§7)."""


class CIMethod(str, Enum):
    """How an interval was computed (§13.3, §13.4)."""

    cluster_bootstrap = "cluster_bootstrap"
    t = "t"
    permutation = "permutation"
    """Exact sign-flip enumeration. Used below the cluster floor, where
    2**n assignments can be listed outright and a bootstrap is forbidden."""

    none = "none"


class Estimand(str, Enum):
    """What an interval covers (§13.2).

    ``generalization`` is primary: obtained by resampling probes, it answers
    "would this hold on a corpus like this one". ``conditional`` holds probes
    fixed and covers only run-to-run stochasticity.
    """

    conditional = "conditional"
    generalization = "generalization"


class MetricValue(BaseModel):
    """A measured value. Cannot exist without an interval or a flagged refusal."""

    model_config = ConfigDict(frozen=True)

    point: float
    lo: float | None = None
    hi: float | None = None
    method: CIMethod
    n_clusters: int
    alpha: float
    estimand: Estimand
    flags: tuple[Flag, ...] = ()

    @model_validator(mode="after")
    def _interval_or_explicit_refusal(self) -> MetricValue:
        if self.method is CIMethod.none:
            if Flag.NO_VALID_INTERVAL not in self.flags:
                raise ValueError(
                    "method=none requires the NO_VALID_INTERVAL flag: a metric may "
                    "decline to give an interval, but never silently (I3)"
                )
            if self.lo is not None or self.hi is not None:
                raise ValueError("method=none must not carry bounds")
            return self

        if self.lo is None or self.hi is None:
            raise ValueError(
                "a MetricValue needs both lo and hi, or method=none with the "
                "NO_VALID_INTERVAL flag (I3)"
            )
        if self.lo > self.hi:
            raise ValueError(f"lo {self.lo} exceeds hi {self.hi}")
        if not (self.lo <= self.point <= self.hi):
            raise ValueError(f"point {self.point} lies outside [{self.lo}, {self.hi}]")
        return self


class MetricSpec(BaseModel):
    """Declaration of a metric: what it measures and how it is resampled.

    ``cluster_key`` names the resampling unit (§13.3). Each objective resamples
    its own cluster set, so this is not incidental metadata — it determines
    which bootstrap a comparison runs.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    family: str
    direction: Literal["maximize", "minimize"]
    unit: str
    cluster_key: str
