"""Golden snapshot of the public API surface (spec §4.2).

The library is public (D32), which taxes every refactor unless "public" is a
deliberate act. This test makes it one: changing a Tier 1 or Tier 2 symbol or
signature fails CI until the snapshot is updated in the same commit, so
widening the surface is visible in review rather than accidental.

Regenerate deliberately with: ``pytest tests/golden --snapshot-update``
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import sweepeval.api as api
from sweepeval.report import Reporter  # noqa: F401  (Tier 2, added by M9)
from sweepeval.schema import call, comparability, metric, objective, observation, unit
from sweepeval.store import blob, jsonl, redaction, run, state

SNAPSHOT = Path(__file__).parent / "api_surface.json"

# Tier 1: frozen for 0.x except additive change.
TIER1 = [api]

# Tier 2: subsystem interfaces. Breaking changes need a CHANGELOG entry and a
# one-minor deprecation shim.
TIER2 = [
    unit,
    metric,
    call,
    observation,
    comparability,
    objective,
    redaction,
    blob,
    jsonl,
    state,
    run,
]


def _signature(obj: object) -> str:
    try:
        return str(inspect.signature(obj))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "<no signature>"


def _surface() -> dict[str, dict[str, str]]:
    surface: dict[str, dict[str, str]] = {}
    for module in TIER1 + TIER2:
        names = getattr(module, "__all__", None)
        if names is None:
            names = [n for n in dir(module) if not n.startswith("_")]
        entries: dict[str, str] = {}
        for name in sorted(names):
            member = getattr(module, name)
            if inspect.isclass(member):
                methods = sorted(
                    m
                    for m in dir(member)
                    if not m.startswith("_") and callable(getattr(member, m, None))
                )
                entries[name] = "class(" + ",".join(methods) + ")"
            elif callable(member):
                entries[name] = "def" + _signature(member)
            else:
                entries[name] = type(member).__name__
        surface[module.__name__] = entries
    return surface


def test_api_surface_matches_the_snapshot(request: pytest.FixtureRequest) -> None:
    current = _surface()

    if request.config.getoption("--snapshot-update", default=False):
        SNAPSHOT.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pytest.skip("snapshot updated")

    if not SNAPSHOT.exists():
        SNAPSHOT.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        pytest.skip("snapshot created")

    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert current == expected, (
        "The public API surface changed. If that was intentional, update the "
        "snapshot in the same commit with: pytest tests/golden --snapshot-update"
    )


def test_every_cli_verb_has_a_tier1_function() -> None:
    """§4.2: "no logic lives only in the CLI"."""
    verbs = {
        "discover",
        "evaluate",
        "sweep",
        "run",
        "baseline",
        "gate",
        "compare",
        "report",
        "demo",
    }
    assert verbs <= set(api.__all__)


def test_gate_and_run_gate_are_distinct() -> None:
    """Rev 1 gave one name two meanings: a pure function and one that re-runs."""
    assert "gate" in api.__all__
    assert "run_gate" in api.__all__
    assert api.gate is not api.run_gate


def test_async_twins_exist_for_the_io_bound_verbs() -> None:
    for name in ("discover", "evaluate", "sweep", "run", "report"):
        assert f"a{name}" in api.__all__, name
        assert inspect.iscoroutinefunction(getattr(api, f"a{name}"))


def test_unimplemented_verbs_say_which_milestone_they_land_in() -> None:
    with pytest.raises(NotImplementedError, match="M2"):
        api.discover("https://x.test")
