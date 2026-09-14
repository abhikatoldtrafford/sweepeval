"""LLM-judge escalation (spec §11.9).

Triggered by a scoring contract returning AMBIGUOUS per its declared
``ambiguous_when``, and by nothing else. Everything a deterministic contract
could settle has already been settled before this package is reached.
"""

from __future__ import annotations

from sweepeval.judge.client import (
    JudgeConfig,
    JudgeError,
    JudgeVerdict,
    check_independence,
    endpoint_fingerprint,
    parse_verdict,
    shares_vendor_prefix,
)
from sweepeval.judge.escalate import (
    AMBIGUOUS_PREFIX,
    Escalation,
    EscalationPlan,
    plan_escalations,
    resolved_observation,
)
from sweepeval.judge.rubric import RUBRIC_VERSION, Rubric, rubric_for

__all__ = [
    "AMBIGUOUS_PREFIX",
    "RUBRIC_VERSION",
    "Escalation",
    "EscalationPlan",
    "JudgeConfig",
    "JudgeError",
    "JudgeVerdict",
    "Rubric",
    "check_independence",
    "endpoint_fingerprint",
    "parse_verdict",
    "plan_escalations",
    "resolved_observation",
    "rubric_for",
    "shares_vendor_prefix",
]
