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
from typing import TYPE_CHECKING, Literal

from sweepeval.corpus.loader import Corpus

if TYPE_CHECKING:  # `cost` imports CHARS_PER_TOKEN from here, so the
    # runtime import would be circular. Only the annotation is needed.
    from sweepeval.execute.cost import Pricing

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
_CAPABILITY_POSTS = 60
_PROBE_TOKENS = 200
"""Ceiling per inert discovery or capability probe.

Both phases send short, fixed prompts. Measured against the mock, discovery
spends 47 tokens across 2 posts; 200 apiece is still an order of magnitude of
headroom, which is what a pre-flight estimate should carry.

It used to be a flat 12,000 for discovery and 200,000 for capabilities -- the
latter 3,333 tokens per request, against 58 per request for the entire scoring
phase. That reservation existed to cover the context-ceiling search, which
runs only under ``--profile deep``, and it made the token cap unusable at
every other profile: `forbids_starting` demanded at least 219,000 tokens, and
a whole sweep then measured about 11,000, so no cap value produced a partial
run. Either it declined the sweep outright or it never bound.
"""

_JUDGE_TOKENS = 800
"""Ceiling per judge call: the rubric, the probe, the response, and a short
JSON verdict back. The phase carried 0 tokens, which made a judged run's
estimate silently exclude the judge's own spend."""

_CONTEXT_CEILING_TOKENS = 200_000
"""The deep-only context-ceiling search, which deliberately sends long inputs
to find where the target truncates. Reserved only for the profile that runs
it (`capabilities/detect.py` skips it otherwise)."""

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
    def unavoidable_tokens(self) -> int:
        """Discovery and capability detection, in tokens.

        The request-side twin of :attr:`unavoidable_requests`, which existed
        while this was recomputed inline in two places that could drift.
        """
        return sum(
            p.tokens
            for p in self.phases
            if p.phase in ("discovery", "capabilities")
        )

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
            unavoidable = estimate.unavoidable_tokens
            per_config = estimate.per_config_tokens
        else:
            unavoidable = estimate.unavoidable_requests
            per_config = estimate.per_config_requests
        return (
            f"{unavoidable:,} for discovery and capability detection plus "
            f"{per_config:,} for one configuration is {unavoidable + per_config:,} "
            f"{self.unit}, above your cap of {self.value:,}"
        )

    def forbids_starting(
        self, estimate: Estimate, pricing: Pricing | None = None
    ) -> bool:
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
            return (
                estimate.unavoidable_tokens + estimate.per_config_tokens > self.value
            )
        if self.unit == "dollars":
            # A dollar cap below the cost of discovery plus one config buys no
            # sweep either -- and returning False here let `BudgetCap(0.01,
            # "dollars")` send 37 requests before coming back with no configs
            # at all. Only decidable with pricing; without it the cap is inert
            # by D8 and the pre-flight has already said so.
            if pricing is None:
                return False
            return (
                pricing.cost(
                    estimate.unavoidable_tokens + estimate.per_config_tokens, 0
                )
                > self.value
            )
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

    ceiling_search = _CONTEXT_CEILING_TOKENS if profile == "deep" else 0
    phases = [
        PhaseEstimate(
            "discovery", _DISCOVERY_POSTS, _DISCOVERY_POSTS * _PROBE_TOKENS, "hard cap"
        ),
        PhaseEstimate(
            "capabilities",
            _CAPABILITY_POSTS,
            _CAPABILITY_POSTS * _PROBE_TOKENS + ceiling_search,
            "hard cap"
            + (
                "; includes the context-ceiling search"
                if ceiling_search
                else "; context-ceiling search runs only under --profile deep"
            ),
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
        # Worst case: every ambiguity-capable unit escalates on every run of
        # every config. It never does -- about 40% of guardrail probes land in
        # the band on live data -- but the pre-flight is what the user
        # consents to under I9, and consenting to an optimistic figure is not
        # consent.
        judge_calls = judge_units * runs * configs
        phases.append(
            PhaseEstimate(
                "judge (worst case)",
                judge_calls,
                judge_calls * _JUDGE_TOKENS,
                "one call per ambiguous probe; charged to the judge endpoint",
            )
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
