"""The ``observations.jsonl`` row (spec §6.1, §6.4). I5 load-bearing.

One row per ``(config_id, unit_id, run_idx, scorer, metric)``. That five-part
key corrects a rev-1 defect: the three-tuple ``(config_id, unit_id, run_idx)``
is not unique, because ``Scorer.score`` returns a list and one context
conversation yields fact recall, constraint persistence and distractor
resistance from the same Unit run.

I5: an unmeasurable scorer reports ``SKIPPED`` with a machine-readable reason.
``Verdict`` has no default and the reason is required, so a silent pass, fail
or zero is not expressible.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field, model_validator

__all__ = ["Layer", "Observation", "ObservationKey", "Verdict"]

ObservationKey = tuple[str, str, int, str, str]

Layer = Literal["generic", "user", "generated"]


class Verdict(str, Enum):
    """No default value, deliberately (I5)."""

    PASS = "PASS"
    FAIL = "FAIL"
    UNSCORABLE = "UNSCORABLE"
    SKIPPED = "SKIPPED"


class Observation(BaseModel):
    """One scored result."""

    model_config = ConfigDict(frozen=True)

    ts: str
    run_id: str
    config_id: str
    unit_id: str
    run_idx: int
    scorer: str
    scorer_version: int
    metric: str
    family: str
    layer: Layer
    verdict: Verdict
    value: float | None = None
    reason: str | None = None
    severity: str | None = None
    attack_class: str | None = None
    policy_id: str | None = None
    depth: int | None = None
    canary_id: str | None = None
    call_ids: tuple[str, ...] = ()
    blob_ids: tuple[str, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def key(self) -> ObservationKey:
        return (self.config_id, self.unit_id, self.run_idx, self.scorer, self.metric)

    @model_validator(mode="after")
    def _reason_required_when_not_scored(self) -> Observation:
        if self.verdict in (Verdict.UNSCORABLE, Verdict.SKIPPED) and not (
            self.reason and self.reason.strip()
        ):
            raise ValueError(
                f"{self.verdict.value} requires a machine-readable reason: every skip "
                "is visible in the report, never silent (I5)"
            )
        return self
