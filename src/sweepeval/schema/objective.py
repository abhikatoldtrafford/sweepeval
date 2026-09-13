"""The objective registry (spec §14.1, §13.6, §11.4, §14.2). I1 and I10 load-bearing.

Data-driven from M0 so that M9's ranking is objective-count-agnostic. If the
ranker hardcoded six objectives, a plugin metric could never reach the frontier
and I10 ("adding a scorer, discovery shape, objective or reporter touches no
runner code") would be false.

**I1** — no *cross-family* composite. Within-family aggregation is legitimate
but must be declared, so ``weighting`` states it and the reporter reads the
declaration rather than recomputing it. ``security_pass_rate`` weights all
eight attack classes equally despite their differing severities; severity
drives hard-fail classification (§11.2), not weighting.

**The two determinism objectives.** ``target_determinism_at_temp0`` is measured
at a pinned ``temp=0`` and is a property of the (model, system_prompt) pair,
not of a full config. ``config_repeatability`` is measured at each config's own
settings and is registered alongside as a non-default, promotable objective.
Registering both is what stops a ``temp=1.0`` row from displaying a
repeatability number taken at ``temp=0`` with nothing beside it to correct the
impression (§11.4, §14.2).
"""

from __future__ import annotations

from collections.abc import Container
from importlib.metadata import entry_points
from typing import Literal

from pydantic import BaseModel, ConfigDict

__all__ = ["OBJECTIVE_ENTRY_POINT_GROUP", "REGISTRY", "Objective", "ObjectiveRegistry"]

OBJECTIVE_ENTRY_POINT_GROUP = "sweepeval.objectives"

Direction = Literal["maximize", "minimize"]
MinEffectKind = Literal["absolute", "relative"]


class Objective(BaseModel):
    """One rankable metric."""

    model_config = ConfigDict(frozen=True)

    id: str
    display_label: str
    direction: Direction
    family: str
    cluster_key: str
    """The resampling unit for this objective's paired bootstrap (§13.3)."""

    min_effect: float
    """Below this difference, configs are treated as equivalent regardless of
    p-value (§13.6). Statistical significance is not importance."""

    min_effect_kind: MinEffectKind

    metric_key: str = ""
    """Which cluster table this objective resamples, when it is not the id.

    Two of the six differ: ``latency_p95_ms`` is stored under ``latency_ms``
    and ``cost_per_probe`` under ``tokens_out``. Looking those up by id found
    nothing, which is how the gate came to silently gate nothing on four of
    six objectives. Declaring it here rather than in a lookup table keeps the
    two consumers -- the frontier and the gate -- from drifting apart, and
    keeps the mapping out of ``stats``, which may not import ``rank``.
    """

    default: bool = False
    note: str = ""
    weighting: str = ""
    """How this metric aggregates within its family, stated for I1."""

    preferred_metric_key: str = ""
    """A better cluster table to use when the run happens to have one.

    ``cost_per_probe`` is the case: with user-supplied pricing the run carries
    a real ``cost_usd`` table, and without it the objective degrades to
    ``tokens_out`` -- which the registry note has always said, and which
    ``pricing_source`` makes a hard comparability key.

    It was a static ``metric_key`` of ``tokens_out``, so the degraded form was
    the *only* form. Supplying prices changed the printed cost block and
    nothing else: the axis labelled "Cost per probe" still ranked output
    tokens, and preferred a reasoning model at $0.021 a probe over a plain one
    at $0.006.
    """

    @property
    def metric(self) -> str:
        """The cluster-table key: ``metric_key`` when set, else the id."""
        return self.metric_key or self.id

    def metric_for(self, available: Container[str]) -> str:
        """The best cluster table this run actually has for the objective."""
        if self.preferred_metric_key and self.preferred_metric_key in available:
            return self.preferred_metric_key
        return self.metric


class ObjectiveRegistry:
    """Registry of objectives. Plugins add to it; ``rank/`` only reads it."""

    def __init__(self) -> None:
        self._objectives: dict[str, Objective] = {}
        self._loaded_plugins = False
        self.plugin_errors: list[tuple[str, str]] = []

    def register(self, objective: Objective) -> None:
        if objective.id in self._objectives:
            raise ValueError(
                f"objective {objective.id!r} is already registered; "
                "re-registering would silently change what a frontier axis means"
            )
        self._objectives[objective.id] = objective

    def get(self, objective_id: str) -> Objective:
        self.from_entry_points()
        try:
            return self._objectives[objective_id]
        except KeyError:
            known = ", ".join(sorted(self._objectives))
            raise KeyError(f"unknown objective {objective_id!r}; known: {known}") from None

    def all(self) -> tuple[Objective, ...]:
        self.from_entry_points()
        return tuple(self._objectives[k] for k in sorted(self._objectives))

    def defaults(self) -> tuple[Objective, ...]:
        return tuple(o for o in self.all() if o.default)

    def from_entry_points(self) -> None:
        """Load third-party objectives declared under ``sweepeval.objectives``.

        Idempotent, and called by :meth:`all` and :meth:`defaults` so a plugin
        is loaded before anything reads the registry. It had no callers, so an
        installed objective was never registered and I10's claim held only
        because adding one did nothing.
        """
        if self._loaded_plugins:
            return
        self._loaded_plugins = True
        for entry in entry_points(group=OBJECTIVE_ENTRY_POINT_GROUP):
            try:
                loaded = entry.load()
                for objective in loaded() if callable(loaded) else loaded:
                    self.register(objective)
            except Exception as error:
                # Arbitrary third-party code. Recorded rather than raised, and
                # surfaced by the CLI -- silence is the defect being fixed.
                self.plugin_errors.append(
                    (entry.name, f"{type(error).__name__}: {error}")
                )


REGISTRY = ObjectiveRegistry()

# The six defaults of §14.1: one per shipped family, except operational, which
# contributes latency and cost separately because they trade off against each
# other.
for _objective in (
    Objective(
        id="security_pass_rate",
        display_label="Security pass rate",
        direction="maximize",
        family="security",
        cluster_key="security_probe",
        min_effect=0.02,
        min_effect_kind="absolute",
        default=True,
        weighting="all 8 attack classes weighted equally; severity drives "
        "hard-fail classification, not weighting (§14.1, I1)",
    ),
    Objective(
        id="guardrail_pass_rate",
        display_label="Guardrail pass rate",
        direction="maximize",
        family="guardrail",
        cluster_key="guardrail_probe",
        min_effect=0.02,
        min_effect_kind="absolute",
        default=True,
        weighting="all 5 policy areas and 4 pressure levels weighted equally",
    ),
    Objective(
        id="target_determinism_at_temp0",
        display_label="Determinism at temp=0",
        direction="maximize",
        family="determinism",
        cluster_key="determinism_base_prompt",
        min_effect=0.02,
        min_effect_kind="absolute",
        default=True,
        note="A property of the (model, system_prompt) pair, measured at a pinned "
        "temp=0 and shared across that pair's temperature siblings. It does not "
        "describe the row's own sampling settings — see config_repeatability for "
        "that (§11.4, §14.2).",
        weighting="unweighted mean over base prompts",
    ),
    Objective(
        id="context_retention_auc",
        display_label="Context retention (AUC)",
        direction="maximize",
        family="context",
        cluster_key="conversation",
        min_effect=0.02,
        min_effect_kind="absolute",
        default=True,
        note="Normalised trapezoid area over the measured depth ladder. Depth "
        "spacing sets the weights, which is why profile is a hard comparability "
        "key (§11.5).",
        weighting="trapezoid over depths; weights published in the report",
    ),
    Objective(
        id="latency_mean_ms",
        metric_key="latency_ms",
        display_label="Latency (mean)",
        direction="minimize",
        family="operational",
        cluster_key="probe",
        min_effect=0.50,
        min_effect_kind="relative",
        default=True,
        note="The MEAN of per-probe latency, not the p95 of §14.1. Measured: "
        "a p95 over 40 clusters establishes non-inferiority on two identical "
        "distributions only 55% of the time even at a 50% margin, and its "
        "bootstrap coverage is 0.88 against a nominal 0.95 -- which is exactly "
        "the condition §13.3 names for falling back. Because domination "
        "requires non-inferiority on EVERY objective, a latency objective that "
        "cannot establish it blocks the entire frontier: a config failing every "
        "security probe stayed non-dominated. The mean reaches 88% at the same "
        "margin and never falsely clears a config that is genuinely 2x slower. "
        "latency_p95_ms is still computed and promotable with --objectives.",
        weighting="mean over probes of per-probe mean call latency; retries and "
        "queue time excluded",
    ),
    Objective(
        id="latency_p95_ms",
        display_label="Latency p95",
        direction="minimize",
        family="operational",
        cluster_key="probe",
        min_effect=0.50,
        min_effect_kind="relative",
        default=False,
        note="The 95th percentile of per-probe mean latency. Reported, and "
        "promotable onto the frontier, but not a default: over a corpus-sized "
        "cluster list it is too noisy to establish non-inferiority, and an "
        "objective that cannot do that blocks every domination (§13.3, §13.5).",
        weighting="quantile over resampled probes; retries and queue time "
        "excluded",
    ),
    Objective(
        id="latency_p90_ms",
        display_label="Latency p90",
        direction="minimize",
        family="operational",
        cluster_key="probe",
        min_effect=0.50,
        min_effect_kind="relative",
        default=False,
        note="§13.3's documented fallback for the p95, now actually emitted. A "
        "distribution-free upper bound needs 72 probes for a p95 and 36 for a "
        "p90, so at the standard profile's 69 probes this is the tail latency "
        "that can carry an honest interval and the p95 is not.",
        weighting="order statistic over probes; retries and queue time excluded",
    ),
    Objective(
        id="cost_per_probe",
        metric_key="tokens_out",
        preferred_metric_key="cost_usd",
        display_label="Cost per probe",
        direction="minimize",
        family="operational",
        cluster_key="probe",
        min_effect=0.10,
        min_effect_kind="relative",
        default=True,
        note="Degrades to tokens_out_per_probe when no pricing is supplied; that "
        "is a different quantity, so pricing_source is a hard comparability key "
        "(§12.4).",
        weighting="mean over probes",
    ),
    # Registered, not default: reported beside the objective above so the
    # production-truth number is visible without distorting the frontier.
    Objective(
        id="config_repeatability",
        display_label="Repeatability at own settings",
        direction="maximize",
        family="determinism",
        cluster_key="determinism_base_prompt",
        min_effect=0.02,
        min_effect_kind="absolute",
        default=False,
        note="Byte-match rate at the config's OWN sampling settings — the "
        "production-truth number. Not a default objective because it is a "
        "deterministic restatement of the temperature axis: every temp=0 config "
        "would score ~1.0 and be automatically non-dominated (§11.4).",
        weighting="unweighted mean over base prompts",
    ),
):
    REGISTRY.register(_objective)

del _objective
