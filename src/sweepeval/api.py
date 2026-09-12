"""Tier 1 public API (spec §4.2).

Additive change only during 0.x. Every CLI verb has a function here and the CLI
calls it, so no logic lives only in the CLI.

``gate`` and ``run_gate`` are separate names because they are separate
operations: ``gate`` is a pure function over an existing result plus a
baseline, while ``run_gate`` re-runs the target first. Rev 1 gave both meanings
to one name.

Milestones fill these in; each raises until its milestone lands, so the surface
is stable from M0 and ``tests/golden/test_api_surface.py`` can pin it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    "adiscover",
    "aevaluate",
    "agate",
    "areport",
    "arun",
    "arun_gate",
    "asweep",
    "baseline",
    "compare",
    "demo",
    "discover",
    "evaluate",
    "gate",
    "report",
    "run",
    "run_gate",
    "sweep",
]

_M = {
    "discover": "M2",
    "evaluate": "M6",
    "baseline": "M6",
    "gate": "M6",
    "run_gate": "M6",
    "demo": "M10",
}


def _pending(name: str) -> NotImplementedError:
    return NotImplementedError(
        f"sweepeval.api.{name}() lands in milestone {_M[name]}; "
        "the signature is stable now so the API surface can be pinned"
    )


def discover(url: str, *, key: str | None = None, **kwargs: Any) -> Any:
    """Phase 0 only: probe the endpoint and emit an annotated config (§8)."""
    import asyncio

    return asyncio.run(adiscover(url, key=key, **kwargs))


def evaluate(url: str, *, key: str | None = None, **kwargs: Any) -> Any:
    """Score a single configuration (§12.1 empty-sweep path)."""
    import asyncio

    return asyncio.run(aevaluate(url, key=key, **kwargs))


def sweep(url: str, *, key: str | None = None, **kwargs: Any) -> Any:
    """Discover, plan and sweep every discoverable configuration (§12)."""
    import asyncio

    return asyncio.run(asweep(url, key=key, **kwargs))


def run(url: str, *, key: str | None = None, **kwargs: Any) -> Any:
    """Zero-config: discover, plan, sweep, rank, report (§1)."""
    import asyncio

    return asyncio.run(arun(url, key=key, **kwargs))


def baseline(result: Any, **kwargs: Any) -> Any:
    """Snapshot a result as a committable baseline (§16)."""
    from sweepeval.execute.gate import snapshot

    return snapshot(result, **kwargs)


def gate(result: Any, *, baseline: Path | str, **kwargs: Any) -> Any:
    """Compare an existing result against a baseline. Pure; sends nothing (§16)."""
    from sweepeval.execute.gate import gate as _gate

    return _gate(result, baseline, **kwargs)


def run_gate(url: str, *, baseline: Path | str, **kwargs: Any) -> Any:
    """Re-run the target, then gate. Distinct from :func:`gate` (§16)."""
    import asyncio

    return asyncio.run(arun_gate(url, baseline=baseline, **kwargs))


def compare(run_a: str, run_b: str, **kwargs: Any) -> Any:
    """Diff two results, refusing invalid comparisons (§6.5, I6). Pure."""
    from sweepeval.report.compare import compare_runs

    return compare_runs(run_a, run_b, **kwargs)


def report(run_id: str, *, fmt: str = "terminal", **kwargs: Any) -> Any:
    """Rebuild a report offline from stored aggregates (§15). Sends nothing."""
    from sweepeval.report.stored import load_run

    return load_run(run_id)


def demo(**kwargs: Any) -> Any:
    """Run against the bundled scenario mock. No URL, no key, no spend (§4.3)."""
    raise _pending("demo")


async def adiscover(url: str, *, key: str | None = None, **kwargs: Any) -> Any:
    """Async twin of :func:`discover`."""
    import httpx

    from sweepeval.discovery.budget import DiscoveryBudget
    from sweepeval.discovery.runner import discover_target

    budget = DiscoveryBudget(max_posts=kwargs.pop("max_posts", 25))
    client = kwargs.pop("client", None)
    if client is not None:
        return await discover_target(client, url, key, budget=budget, **kwargs)
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as owned:
        return await discover_target(owned, url, key, budget=budget, **kwargs)


async def aevaluate(url: str, *, key: str | None = None, **kwargs: Any) -> Any:
    """Async twin of :func:`evaluate`."""
    from sweepeval.execute.evaluate import aevaluate_target

    return await aevaluate_target(url, key=key, **kwargs)


async def asweep(url: str, *, key: str | None = None, **kwargs: Any) -> Any:
    """Async twin of :func:`sweep`."""
    from sweepeval.execute.sweep import asweep_target

    return await asweep_target(url, key=key, **kwargs)


async def arun(url: str, *, key: str | None = None, **kwargs: Any) -> Any:
    """Async twin of :func:`run`. The zero-config path is a sweep (§12.1)."""
    return await asweep(url, key=key, **kwargs)


async def agate(result: Any, *, baseline: Path | str, **kwargs: Any) -> Any:
    """Async twin of :func:`gate`. Pure, so it merely defers."""
    return gate(result, baseline=baseline, **kwargs)


async def arun_gate(url: str, *, baseline: Path | str, **kwargs: Any) -> Any:
    """Async twin of :func:`run_gate`."""
    from sweepeval.execute.gate import gate as _gate

    gate_kwargs = {
        k: kwargs.pop(k)
        for k in ("alpha", "gate_on", "min_effect_overrides", "objectives")
        if k in kwargs
    }
    result = await aevaluate(url, **kwargs)
    return _gate(result, baseline, **gate_kwargs)


async def areport(run_id: str, *, fmt: str = "terminal", **kwargs: Any) -> Any:
    """Async twin of :func:`report`. Pure, so it merely defers."""
    return report(run_id, fmt=fmt, **kwargs)
