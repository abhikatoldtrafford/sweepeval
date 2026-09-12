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
import types
from enum import Enum
from pathlib import Path
from typing import get_origin

import pytest
from pydantic import BaseModel

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


def _describe_class(cls: type) -> str:
    """Describe a class by what *we* declared, not by what it inherits.

    ``dir()`` is not portable across Python versions — enums and pydantic
    models gain and lose inherited members between 3.10 and 3.12 — so a
    snapshot built from it fails CI on the version it was not generated under,
    which says nothing about our API.
    """
    if issubclass(cls, Enum):
        return "enum(" + ",".join(m.name for m in cls) + ")"

    if issubclass(cls, BaseModel):
        fields = ",".join(sorted(cls.model_fields))
        own = ",".join(
            sorted(
                name
                for name, value in vars(cls).items()
                if not name.startswith("_") and callable(value)
            )
        )
        return f"model(fields={fields};methods={own})"

    own_methods = ",".join(
        sorted(
            name
            for name, value in vars(cls).items()
            if not name.startswith("_") and callable(value)
        )
    )
    own_props = ",".join(
        sorted(
            name
            for name, value in vars(cls).items()
            if not name.startswith("_") and isinstance(value, property)
        )
    )
    return f"class(methods={own_methods};props={own_props})"


def _surface() -> dict[str, dict[str, str]]:
    surface: dict[str, dict[str, str]] = {}
    for module in TIER1 + TIER2:
        names = getattr(module, "__all__", None)
        if names is None:
            names = [n for n in dir(module) if not n.startswith("_")]
        entries: dict[str, str] = {}
        for name in sorted(names):
            member = getattr(module, name)
            if isinstance(member, types.GenericAlias) or get_origin(member) is not None:
                # A type alias such as ObservationKey = tuple[...]. 3.10 sees a
                # GenericAlias as a class and 3.12 sees it as callable, so
                # render it by its string form, which is stable on both.
                entries[name] = f"alias({member})"
            elif inspect.isclass(member):
                entries[name] = _describe_class(member)
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
    """Verbs whose milestone has not landed name it, rather than failing
    with an opaque AttributeError or a silent no-op."""
    with pytest.raises(NotImplementedError, match="M6"):
        api.evaluate("https://x.test")
    with pytest.raises(NotImplementedError, match="M8"):
        api.sweep()
    with pytest.raises(NotImplementedError, match="M9"):
        api.report("run-1")


def test_discover_is_implemented() -> None:
    """M2 landed, so discover must no longer raise NotImplementedError."""
    import inspect

    assert "raise _pending" not in inspect.getsource(api.discover)
