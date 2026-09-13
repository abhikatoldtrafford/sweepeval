"""I10: adding a scorer or objective touches no runner code.

The invariant table names four mechanisms. `from_entry_points` implemented two
of them -- one on `ScorerRegistry`, one on `ObjectiveRegistry` -- and had
**zero callers**, in the package or the tests:

    $ grep -rn from_entry_points --include=*.py src tests
    src/sweepeval/schema/objective.py:105:    def from_entry_points(self) -> None:
    src/sweepeval/scorers/base.py:200:    def from_entry_points(self) -> None:

So an installed third-party scorer was never registered, and I10 held only in
the sense that adding one did nothing at all. The units of an unregistered
family then hit `except KeyError: return observations` in the runner and
vanished: executed, paid for, and absent from the report with no SKIPPED row
to say so.

This is the same shape as the I7 provenance gap -- a mechanism fully written,
tested in isolation, and connected to nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sweepeval.capabilities.detect import Capability
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.objective import REGISTRY, Objective, ObjectiveRegistry
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import ScoringContract, Turn, Unit
from sweepeval.scorers.base import ScorerRegistry, registry


@dataclass
class _PluginScorer:
    family: str = "retrieval_quality"
    version: int = 1
    requires: frozenset[Capability] = field(default_factory=frozenset)

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                metric="retrieval_quality", family=self.family,
                direction="maximize", unit="rate", cluster_key="probe",
            )
        ]

    def score(self, unit, calls, context) -> list[Observation]:
        return [
            context.observation(
                scorer=self.family, version=self.version,
                metric="retrieval_quality", family=self.family,
                verdict=Verdict.PASS, value=1.0, reason="plugin", unit=unit,
            )
        ]


class _FakeEntry:
    def __init__(self, name, value):
        self.name = name
        self._value = value

    def load(self):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


# --- the loaders are actually called --------------------------------------


def test_reading_the_scorer_registry_loads_plugins(monkeypatch) -> None:
    """`registry()` is the only way the runner obtains scorers, so that is
    where loading has to happen -- not in a method nobody calls."""
    fresh = ScorerRegistry()
    monkeypatch.setattr(
        "sweepeval.scorers.base.entry_points",
        lambda group: [_FakeEntry("plug", _PluginScorer)] if group else [],
    )
    fresh.from_entry_points()
    assert fresh.get("retrieval_quality").family == "retrieval_quality"


def test_reading_the_objective_registry_loads_plugins(monkeypatch) -> None:
    fresh = ObjectiveRegistry()
    extra = Objective(
        id="plugin_objective", display_label="Plugin", direction="maximize",
        family="retrieval_quality", cluster_key="probe", min_effect=0.05,
        min_effect_kind="absolute",
    )
    monkeypatch.setattr(
        "sweepeval.schema.objective.entry_points",
        lambda group: [_FakeEntry("plug", lambda: [extra])] if group else [],
    )
    assert any(o.id == "plugin_objective" for o in fresh.all())


def test_loading_is_idempotent(monkeypatch) -> None:
    """`all()` is called many times per run. Re-registering would raise, since
    the registry refuses a duplicate id on purpose."""
    fresh = ScorerRegistry()
    calls = {"n": 0}

    def entries(group):
        calls["n"] += 1
        return [_FakeEntry("plug", _PluginScorer)]

    monkeypatch.setattr("sweepeval.scorers.base.entry_points", entries)
    fresh.from_entry_points()
    fresh.from_entry_points()
    assert calls["n"] == 1


def test_the_shipped_registries_still_load_cleanly() -> None:
    """With no plugins installed, nothing changes and nothing errors."""
    assert registry().plugin_errors == []
    assert REGISTRY.plugin_errors == []
    assert len(REGISTRY.all()) >= 6


# --- a broken plugin is reported, not swallowed ---------------------------


def test_a_plugin_that_fails_to_import_is_recorded(monkeypatch) -> None:
    """Arbitrary third-party code must not make the tool unusable -- but an
    entry point that raises is indistinguishable from one never installed,
    and the user is the only person who can fix either."""
    fresh = ScorerRegistry()
    monkeypatch.setattr(
        "sweepeval.scorers.base.entry_points",
        lambda group: [_FakeEntry("broken", ImportError("no module named foo"))],
    )
    fresh.from_entry_points()
    assert fresh.plugin_errors
    name, reason = fresh.plugin_errors[0]
    assert name == "broken"
    assert "ImportError" in reason


def test_a_broken_plugin_does_not_stop_the_others(monkeypatch) -> None:
    fresh = ScorerRegistry()
    monkeypatch.setattr(
        "sweepeval.scorers.base.entry_points",
        lambda group: [
            _FakeEntry("broken", ImportError("boom")),
            _FakeEntry("good", _PluginScorer),
        ],
    )
    fresh.from_entry_points()
    assert fresh.get("retrieval_quality")
    assert len(fresh.plugin_errors) == 1


# --- and an unscored family says so ---------------------------------------


def test_a_unit_with_no_registered_scorer_reports_SKIPPED() -> None:
    """It returned an empty list, so the probe was executed, paid for, and
    then absent from the report -- indistinguishable from a family that was
    never in the corpus. I5 forbids exactly that."""
    from sweepeval.execute.runner import RunPlan, _score

    unit = Unit.make(
        template_id="unknown.x.v1", family="a_family_with_no_scorer",
        turns=[Turn(role="user", text="hi")], profiles={"quick"},
    )
    plan = RunPlan(config_id="c", units=(unit,), runs=1, master_seed="s")
    rows = _score(
        unit, [], "some response", {}, plan, 0, registry(), failed=False, reason=""
    )
    scored = [o for o in rows if o.family == unit.family]
    assert scored, "the family vanished from the observation log"
    assert scored[0].verdict is Verdict.SKIPPED
    assert "no scorer is registered" in (scored[0].reason or "")


def test_a_registered_family_is_unaffected() -> None:
    """Otherwise the branch above could fire for everything."""
    from sweepeval.execute.runner import RunPlan, _score

    unit = Unit.make(
        template_id="sec.x.v1", family="security",
        turns=[Turn(role="user", text="hi")], profiles={"quick"},
        attack_class="instruction_override",
        scoring=[ScoringContract(kind="canary_absent", canary="primary")],
    )
    plan = RunPlan(config_id="c", units=(unit,), runs=1, master_seed="s")
    rows = _score(
        unit, [], "an answer", {"primary": "7BQ2XKM9DF"}, plan, 0, registry(),
        failed=False, reason="",
    )
    security = [o for o in rows if o.family == "security"]
    assert security and security[0].verdict is not Verdict.SKIPPED
