"""Token accounting and cost (spec §12.4, D8).

**No price table ships with this tool.** Prices change weekly, a stale table
would produce confidently wrong dollar figures, and a wrong number carries
more authority than no number. ``tests/contract/test_no_price_table.py``
greps the repo for one.

Three states, in decreasing order of trust:

``MEASURED``
    The target reported token counts and we used them.
``ESTIMATED``
    It did not, so tokens are ``chars / 4`` — a disclosed heuristic, named in
    the manifest beside every number derived from it.
degraded
    No pricing was supplied, so ``cost_per_probe`` becomes
    ``tokens_out_per_probe``. That is a different quantity in different units,
    which is why ``pricing_source`` is a *hard* comparability key: silently
    swapping one for the other would let two runs be compared as though they
    measured the same thing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from sweepeval.execute.budget import CHARS_PER_TOKEN
from sweepeval.schema.call import Call, ErrorClass

__all__ = [
    "COST_METRIC",
    "DEGRADED_METRIC",
    "MEASURED_THRESHOLD",
    "CostAccounting",
    "Pricing",
    "account",
    "estimate_tokens",
]

COST_METRIC = "cost_per_probe"
DEGRADED_METRIC = "tokens_out_per_probe"

MEASURED_THRESHOLD = 0.95
"""Provenance is per run, not per call: a target that reports usage on most
calls and omits it on a few would otherwise produce a metric that is part
measurement and part heuristic with one label on it. Below this share the
whole figure is ESTIMATED and says so."""

TokenSource = Literal["MEASURED", "ESTIMATED"]


@dataclass(frozen=True)
class Pricing:
    """User-supplied prices, in currency units per million tokens.

    Supplied by the user or read from their config. Never bundled, never
    fetched: a price this tool invented is the one number a user cannot check
    against their own invoice.
    """

    input_per_mtok: float
    output_per_mtok: float
    currency: str = "USD"
    source: str = "user"

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (
            tokens_in * self.input_per_mtok + tokens_out * self.output_per_mtok
        ) / 1_000_000


@dataclass(frozen=True)
class CostAccounting:
    """What one config's calls cost, and how well that is known."""

    probes: int
    tokens_in: int
    tokens_out: int
    reasoning_tokens: int
    source: TokenSource
    pricing_source: str
    metric: str
    """``cost_per_probe`` with pricing, ``tokens_out_per_probe`` without."""

    value_per_probe: float
    currency: str | None = None
    heuristic: str | None = None
    """Named in the manifest whenever ``source`` is ESTIMATED."""

    @property
    def degraded(self) -> bool:
        return self.metric == DEGRADED_METRIC

    def describe(self) -> str:
        if self.degraded:
            return (
                f"{self.value_per_probe:,.0f} output tokens per probe "
                f"({self.source.lower()}; no pricing supplied, so the cost "
                f"objective is tokens, not money)"
            )
        return (
            f"{self.value_per_probe:.6f} {self.currency} per probe "
            f"({self.source.lower()} tokens, pricing from {self.pricing_source})"
        )


def estimate_tokens(text: str) -> int:
    """The disclosed ``chars / 4`` heuristic (§12.4 tier 2)."""
    return len(text) // CHARS_PER_TOKEN


def account(
    calls: Iterable[Call],
    *,
    pricing: Pricing | None = None,
    probe_texts: Sequence[str] = (),
) -> CostAccounting:
    """Total one config's tokens and turn them into a per-probe figure.

    ``probe_texts`` supplies the request text for the ESTIMATED path, since a
    ``Call`` records the request's byte length but not its characters. When it
    is absent the input side falls back to the recorded request bytes, which
    over-counts multi-byte text and is labelled ESTIMATED either way.
    """
    counted = [
        c
        for c in calls
        if c.attempt == 1 and c.response.error_class is not ErrorClass.terminal
    ]

    reported_out = sum(1 for c in counted if c.tokens.out is not None)
    share = reported_out / len(counted) if counted else 0.0
    measured = bool(counted) and share >= MEASURED_THRESHOLD

    if measured:
        tokens_in = sum(c.tokens.in_ or 0 for c in counted)
        tokens_out = sum(c.tokens.out or 0 for c in counted)
        reasoning = sum(c.tokens.reasoning or 0 for c in counted)
        source: TokenSource = "MEASURED"
        heuristic = None
    else:
        tokens_in = (
            sum(estimate_tokens(t) for t in probe_texts)
            if probe_texts
            else sum(c.request.bytes // CHARS_PER_TOKEN for c in counted)
        )
        tokens_out = sum(c.response.bytes // CHARS_PER_TOKEN for c in counted)
        reasoning = 0
        source = "ESTIMATED"
        heuristic = f"chars/{CHARS_PER_TOKEN}"

    # Probes, not calls: a depth-15 conversation is one probe and fifteen
    # calls, and dividing by calls would make multi-turn units look cheap.
    probes = len({(c.unit_id, c.run_idx) for c in counted})

    if pricing is None:
        return CostAccounting(
            probes=probes,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            reasoning_tokens=reasoning,
            source=source,
            pricing_source="none",
            metric=DEGRADED_METRIC,
            value_per_probe=(tokens_out / probes) if probes else 0.0,
            currency=None,
            heuristic=heuristic,
        )

    # Reasoning tokens are billed as output on every provider that reports
    # them separately, so they belong in the cost even though they are not in
    # the answer (§7).
    total = pricing.cost(tokens_in, tokens_out + reasoning)
    return CostAccounting(
        probes=probes,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        reasoning_tokens=reasoning,
        source=source,
        pricing_source=pricing.source,
        metric=COST_METRIC,
        value_per_probe=(total / probes) if probes else 0.0,
        currency=pricing.currency,
        heuristic=heuristic,
    )
