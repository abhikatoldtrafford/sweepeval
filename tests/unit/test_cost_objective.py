"""The cost objective has to rank cost (spec §12.4, D8).

`cost_per_probe` carried a static `metric_key="tokens_out"`, so the degraded
form was the only form. Supplying prices changed the printed cost block and
nothing else: the axis labelled "Cost per probe" still resampled output
tokens, and input and reasoning tokens were invisible to it entirely.

The audit's reproduction -- a reasoning model that answers tersely against a
plain model that answers at length:

    A   100 output tokens, $0.021 a probe
    B   500 output tokens, $0.0051 a probe
    -> the frontier dominated B, preferring the configuration costing 4x more

Degrading to output tokens *without* pricing is correct and documented, and
`pricing_source` is a hard comparability key precisely so the two forms are
never mixed. This file asserts both halves, because a fix that always used
dollars would invent a price the user never gave -- which is what D8 forbids.
"""

from __future__ import annotations

import pytest

from sweepeval.execute.cost import Pricing
from sweepeval.rank.frontier import rank_configs
from sweepeval.schema.metric import CIMethod, Estimand, MetricValue
from sweepeval.schema.objective import REGISTRY

COST = REGISTRY.get("cost_per_probe")
UNITS = [f"u{i}" for i in range(12)]
PRICING = Pricing(input_per_mtok=1.0, output_per_mtok=10.0, source="user")


def _mv(point: float) -> MetricValue:
    return MetricValue(
        point=point, lo=point * 0.9, hi=point * 1.1,
        method=CIMethod.cluster_bootstrap, n_clusters=len(UNITS), alpha=0.05,
        estimand=Estimand.generalization,
    )


def _configs(with_pricing: bool):
    """A is terse but reasons heavily; B is verbose but cheap. Output tokens
    and dollars rank them in opposite orders, which is the whole point."""
    spend = {"A": (20_000, 100), "B": (100, 500)}
    clusters, metrics = {}, {}
    for name, (tin, tout) in spend.items():
        tables = {"tokens_out": {u: float(tout) for u in UNITS}}
        values = {"tokens_out": _mv(float(tout))}
        if with_pricing:
            cost = PRICING.cost(tin, tout)
            tables["cost_usd"] = {u: cost for u in UNITS}
            values["cost_usd"] = _mv(cost)
        clusters[name], metrics[name] = tables, values
    return clusters, metrics


# --- with pricing, the axis is dollars ------------------------------------


def test_the_cheaper_configuration_wins_when_prices_are_supplied() -> None:
    clusters, metrics = _configs(with_pricing=True)
    result = rank_configs(
        ["A", "B"], metrics, clusters, [COST], constraints=(), seed=7
    )
    assert "A" in result.dominated, "the $0.021 config was not dominated"
    assert result.frontier == ("B",)


def test_without_the_fix_output_tokens_rank_them_the_other_way() -> None:
    """Not an aspiration: this is what shipped. The test above is only
    meaningful because the two orderings genuinely disagree."""
    clusters, metrics = _configs(with_pricing=True)
    for tables in clusters.values():
        tables.pop("cost_usd")
    for values in metrics.values():
        values.pop("cost_usd")
    result = rank_configs(
        ["A", "B"], metrics, clusters, [COST], constraints=(), seed=7
    )
    assert "B" in result.dominated


def test_input_and_reasoning_tokens_reach_the_price() -> None:
    """A reasoning model's spend is mostly tokens that never appear in the
    answer -- gpt-5-nano burned 576 of them for a one-line refusal."""
    assert PRICING.cost(20_000, 100) > PRICING.cost(100, 500)


# --- without pricing, it degrades and says so -----------------------------


def test_the_objective_degrades_to_output_tokens_with_no_prices() -> None:
    """D8: a price this tool invented is the one number a user cannot check
    against their own invoice."""
    assert COST.metric_for({"tokens_out", "latency_ms"}) == "tokens_out"


def test_the_objective_prefers_real_cost_when_the_run_has_it() -> None:
    assert COST.metric_for({"tokens_out", "cost_usd"}) == "cost_usd"


def test_an_objective_without_a_preferred_table_is_unaffected() -> None:
    """`metric_for` must not become a second mapping that drifts from
    `metric`."""
    for objective in REGISTRY.all():
        if not objective.preferred_metric_key:
            assert objective.metric_for({"anything"}) == objective.metric


# --- the two sides of a comparison agree on the quantity ------------------


def test_a_pair_is_never_compared_on_different_tables() -> None:
    """One config priced and one not would otherwise compare dollars against
    token counts. Resolution is against what BOTH sides carry."""
    clusters, metrics = _configs(with_pricing=True)
    clusters["A"].pop("cost_usd")
    metrics["A"].pop("cost_usd")

    shared = set(clusters["A"]) & set(clusters["B"])
    assert COST.metric_for(shared) == "tokens_out"

    result = rank_configs(
        ["A", "B"], metrics, clusters, [COST], constraints=(), seed=7
    )
    assert result.frontier, "the pair became uncomparable instead of degrading"


# --- the scorer feeds it ---------------------------------------------------


def test_the_operational_scorer_emits_the_token_families_cost_needs() -> None:
    from sweepeval.schema.call import Call
    from sweepeval.schema.unit import Turn, Unit
    from sweepeval.scorers.base import ScoreContext
    from sweepeval.scorers.operational import OperationalScorer

    unit = Unit.make(
        template_id="op.x.v1", family="security",
        turns=[Turn(role="user", text="hi")], profiles={"quick"},
    )
    context = ScoreContext(
        run_id="r", config_id="c", run_idx=0, text="hello", ts="2026-09-13T00:00:00Z"
    )
    rows = OperationalScorer().score(unit, [Call.example()], context)
    emitted = {o.metric for o in rows}
    assert {"tokens_out", "tokens_in", "tokens_reasoning"} <= emitted, emitted


def test_only_billable_calls_count_toward_the_token_metrics() -> None:
    """`cost.account` bills first attempts that did not terminate. The token
    observations summed every call, so retries and dead requests inflated the
    ranked axis while the printed cost block excluded them."""
    import dataclasses

    from sweepeval.schema.call import Call, ErrorClass
    from sweepeval.schema.unit import Turn, Unit
    from sweepeval.scorers.base import ScoreContext
    from sweepeval.scorers.operational import OperationalScorer

    first = Call.example()
    retry = dataclasses.replace(first, attempt=2) if dataclasses.is_dataclass(
        first
    ) else first.model_copy(update={"attempt": 2})
    dead = first.model_copy(
        update={"response": first.response.model_copy(
            update={"error_class": ErrorClass.terminal}
        )}
    )

    unit = Unit.make(
        template_id="op.x.v1", family="security",
        turns=[Turn(role="user", text="hi")], profiles={"quick"},
    )
    context = ScoreContext(
        run_id="r", config_id="c", run_idx=0, text="hello", ts="2026-09-13T00:00:00Z"
    )
    scorer = OperationalScorer()

    def tokens_out(calls):
        rows = scorer.score(unit, calls, context)
        return next(o.value for o in rows if o.metric == "tokens_out")

    alone = tokens_out([first])
    assert tokens_out([first, retry]) == pytest.approx(alone)
    assert tokens_out([first, dead]) == pytest.approx(alone)
