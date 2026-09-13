"""Pre-flight budget estimate and cap (spec §12.3, §12.4). I9 load-bearing.

I9: "No billable request of any kind is sent before an estimate has been shown
and confirmed."

An earlier revision gated at the planner — after discovery and capability
detection had already spent, including a context-ceiling binary search that can
cost more than the entire scoring sweep. The estimate here covers **every**
phase, and the gate runs before the first request of any kind.

The scoring line sums ``calls_per_run`` over units. ``configs x units x runs``
is wrong for every multi-turn unit, and it was the figure an earlier revision
asked the user to consent to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from sweepeval.corpus.loader import Corpus

__all__ = [
    "CHARS_PER_TOKEN",
    "BudgetCap",
    "Estimate",
    "PhaseEstimate",
    "estimate_run",
    "render_estimate",
]

CHARS_PER_TOKEN = 4
"""§12.4 tier 2. Disclosed, and named in the manifest, because an estimate
presented as a measurement is worse than no estimate."""

_DISCOVERY_POSTS = 25
_DISCOVERY_TOKENS = 12_000
_CAPABILITY_POSTS = 60
_CAPABILITY_TOKENS = 200_000
_HARD_FAIL_CONFIRMATIONS = 3


@dataclass(frozen=True)
class PhaseEstimate:
    phase: str
    requests: int
    tokens: int
    note: str = ""


@dataclass
class Estimate:
    phases: list[PhaseEstimate] = field(default_factory=list)
    profile: str = "quick"
    configs: int = 1
    runs: int = 3
    concurrency: int = 2
    pricing_source: str = "none"
    gate_eligible: bool = True

    @property
    def total_requests(self) -> int:
        return sum(p.requests for p in self.phases)

    @property
    def total_tokens(self) -> int:
        return sum(p.tokens for p in self.phases)

    @property
    def unavoidable_requests(self) -> int:
        """Discovery and capability detection.

        Nothing can be measured without them, so a cap below this figure does
        not buy a smaller sweep — it buys no sweep, and the run declines
        before sending anything rather than spending the budget on discovery
        and then having nothing left to score with.
        """
        return sum(
            p.requests for p in self.phases if p.phase in ("discovery", "capabilities")
        )

    @property
    def per_config_requests(self) -> int:
        """What one more configuration costs, for projecting against a cap."""
        scoring = next(
            (p.requests for p in self.phases if p.phase.startswith("scoring")), 0
        )
        return scoring // max(1, self.configs)

    @property
    def per_config_tokens(self) -> int:
        scoring = next(
            (p.tokens for p in self.phases if p.phase.startswith("scoring")), 0
        )
        return scoring // max(1, self.configs)

    @property
    def wall_clock_minutes(self) -> float:
        """At roughly 2.5s per request, divided by concurrency."""
        return (self.total_requests * 2.5) / max(1, self.concurrency) / 60.0


@dataclass
class BudgetCap:
    """A hard cap in one unit (§12.3)."""

    value: float | None = None
    unit: Literal["requests", "tokens", "dollars"] = "requests"

    def shortfall(self, estimate: Estimate) -> str:
        """Why the cap cannot buy even one configuration, in its own units."""
        if self.unit == "tokens":
            unavoidable = sum(
                p.tokens
                for p in estimate.phases
                if p.phase in ("discovery", "capabilities")
            )
            per_config = estimate.per_config_tokens
        else:
            unavoidable = estimate.unavoidable_requests
            per_config = estimate.per_config_requests
        return (
            f"{unavoidable:,} for discovery and capability detection plus "
            f"{per_config:,} for one configuration is {unavoidable + per_config:,} "
            f"{self.unit}, above your cap of {self.value:,}"
        )

    def forbids_starting(self, estimate: Estimate) -> bool:
        """Whether the cap makes even one configuration impossible (§12.3).

        Distinct from :meth:`exceeded_by`. A cap below the *estimate* is a
        deliberate request for a partial sweep — the runner stops between
        configs and says so. A cap below the *unavoidable* phases is a request
        for something that cannot happen at all.
        """
        if self.value is None:
            return False
        if self.unit == "requests":
            # Room for the unavoidable phases AND at least one configuration.
            # A cap that affords discovery but not one scored config buys no
            # sweep, and spending it on discovery leaves nothing to score.
            return (
                estimate.unavoidable_requests + estimate.per_config_requests
                > self.value
            )
        if self.unit == "tokens":
            unavoidable = sum(
                p.tokens
                for p in estimate.phases
                if p.phase in ("discovery", "capabilities")
            )
            return unavoidable + estimate.per_config_tokens > self.value
        return False

    def exceeded_by(self, estimate: Estimate) -> bool:
        if self.value is None:
            return False
        if self.unit == "requests":
            return estimate.total_requests > self.value
        if self.unit == "tokens":
            return estimate.total_tokens > self.value
        # Dollars need pricing; with none supplied the cap cannot bind, and
        # silently treating "no pricing" as "no cost" would be worse. The
        # pre-flight says so in the same breath as reporting tokens only.
        return False


def estimate_run(
    corpus: Corpus,
    *,
    configs: int,
    runs: int,
    profile: str,
    concurrency: int = 2,
    hard_fail_units: int = 0,
    judge_units: int = 0,
    pricing_source: str = "none",
) -> Estimate:
    """Every phase, before the first request of any kind (I9)."""
    scoring_requests = corpus.calls_per_run * runs * configs
    scoring_tokens = _tokens_for(corpus, runs, configs)

    phases = [
        PhaseEstimate("discovery", _DISCOVERY_POSTS, _DISCOVERY_TOKENS, "hard cap"),
        PhaseEstimate(
            "capabilities", _CAPABILITY_POSTS, _CAPABILITY_TOKENS,
            "hard cap; context-ceiling search runs only under --profile deep",
        ),
        PhaseEstimate(
            f"scoring ({profile})", scoring_requests, scoring_tokens,
            f"{corpus.unit_count} units x {corpus.calls_per_run} calls/run "
            f"x {runs} runs x {configs} config(s)",
        ),
    ]

    if hard_fail_units:
        phases.append(
            PhaseEstimate(
                "hard-fail confirm",
                _HARD_FAIL_CONFIRMATIONS * hard_fail_units * configs,
                0,
                "usually unspent (§11.2)",
            )
        )
    if judge_units:
        phases.append(
            PhaseEstimate("judge (worst case)", judge_units * runs * configs, 0, "")
        )

    return Estimate(
        phases=phases,
        profile=profile,
        configs=configs,
        runs=runs,
        concurrency=concurrency,
        pricing_source=pricing_source,
        gate_eligible=profile != "quick",
    )


def _tokens_for(corpus: Corpus, runs: int, configs: int) -> int:
    """Disclosed chars/4 estimate over the corpus's own prompt text."""
    prompt_chars = sum(
        len(turn.text) for template in corpus.probes for turn in template.turns
    )
    # Prompt tokens grow with the conversation on replay, so a multi-turn unit
    # sends its earlier turns again on every later turn. Approximated by the
    # triangular factor rather than ignored, because context probes dominate
    # the token bill at `standard`.
    replay_factor = 2
    per_run = (prompt_chars * replay_factor) // CHARS_PER_TOKEN
    return per_run * runs * configs


def render_estimate(estimate: Estimate) -> str:
    """The block shown before spending (§12.3).

    Profile, request count, token estimate and wall-clock, in that order, as
    the first thing the user reads.
    """
    width = max(len(p.phase) for p in estimate.phases) + 2
    lines = [
        f"  profile        {estimate.profile}, {estimate.configs} config(s), "
        f"{estimate.runs} runs"
        + ("" if estimate.gate_eligible else " — not gate-eligible"),
        "",
        f"  {'phase':<{width}} {'requests':>10} {'tokens (est)':>14}",
    ]
    for phase in estimate.phases:
        lines.append(
            f"  {phase.phase:<{width}} {phase.requests:>10,} {phase.tokens:>14,}"
            + (f"   {phase.note}" if phase.note else "")
        )
    lines += [
        f"  {'-' * (width + 26)}",
        f"  {'total':<{width}} {estimate.total_requests:>10,} "
        f"{estimate.total_tokens:>14,}",
        "",
    ]

    if estimate.pricing_source == "none":
        lines.append(
            "  cost           no pricing supplied — reporting tokens only, and "
            "the cost objective degrades to tokens_out_per_probe"
        )
    lines.append(f"  wall-clock     ~{estimate.wall_clock_minutes:.0f} min at "
                 f"concurrency {estimate.concurrency}")
    return "\n".join(lines)
