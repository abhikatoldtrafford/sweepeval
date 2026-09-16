"""Scorer protocol and registry (spec §11.1, I10).

I10: adding a scorer touches no runner code. Registration is by decorator
in-tree and by ``entry_points`` out-of-tree, and both go through the same
protocol.

The acceptance test for this interface is a test that registers a fake scorer
and runs it end to end. An interface that has only ever been used by its own
authors is not an interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

from sweepeval.capabilities.detect import Capability, CapabilityReport
from sweepeval.schema.call import Call
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit

__all__ = [
    "SCORER_ENTRY_POINT_GROUP",
    "CrossRunScorer",
    "RunEvidence",
    "ScoreContext",
    "Scorer",
    "ScorerRegistry",
    "register",
    "registry",
]

SCORER_ENTRY_POINT_GROUP = "sweepeval.scorers"


@dataclass
class ScoreContext:
    """Everything a scorer needs that is not the unit or its calls."""

    run_id: str
    config_id: str
    run_idx: int
    text: str
    """Extracted response text of the final turn."""

    texts: Sequence[str] = ()
    """Extracted text of every turn, for multi-turn units."""

    canaries: dict[str, str] = field(default_factory=dict)
    refusal_detected: bool = False
    text_blob_id: str | None = None
    """Content address of :attr:`text`, so an Observation can point at the
    exact bytes it scored.

    Without it a stored run cannot be re-scored: the blobs are on disk and
    nothing says which observation came from which. That turns every scorer
    fix into another paid run against the target, which is the opposite of
    what an append-only artifact store is for.
    """

    payload: Any = None
    """The final response, parsed, when a scorer needs its structure.

    Everything else here scores *text*, which is all the extraction path
    returns. Tool calls are not text: they live in `tool_calls`,
    `function_call` or a `tool_use` block, and a target that emits only a call
    returns a null `content` -- so `text` is empty exactly when there is most
    to score. Carried rather than re-read from the blob store so the scorer
    sees the same bytes the run did.
    """

    layer: str = "generic"
    ts: str = ""

    def observation(
        self,
        *,
        scorer: str,
        version: int,
        metric: str,
        family: str,
        verdict: Verdict,
        value: float | None = None,
        reason: str | None = None,
        unit: Unit | None = None,
        call_ids: Sequence[str] = (),
        blob_ids: Sequence[str] = (),
    ) -> Observation:
        """Build an Observation with the run/config/unit keys already filled."""
        return Observation(
            ts=self.ts,
            run_id=self.run_id,
            config_id=self.config_id,
            unit_id=unit.unit_id if unit else "",
            run_idx=self.run_idx,
            scorer=scorer,
            scorer_version=version,
            metric=metric,
            family=family,
            layer=self.layer,  # type: ignore[arg-type]
            verdict=verdict,
            value=value,
            reason=reason,
            severity=unit.severity if unit else None,
            attack_class=unit.attack_class if unit else None,
            policy_id=unit.policy_id if unit else None,
            depth=unit.depth if unit else None,
            call_ids=tuple(call_ids),
            # Defaults to the address of the text that was scored, so a
            # stored run can be re-scored after a scorer fix instead of
            # costing another paid run against the target.
            blob_ids=tuple(blob_ids)
            or ((self.text_blob_id,) if self.text_blob_id else ()),
        )


@dataclass
class RunEvidence:
    """One unit's text across every run, for cross-run scorers."""

    unit: Unit
    texts: tuple[str, ...]
    """Extracted final-turn text, indexed by ``run_idx``."""

    unscorable: tuple[int, ...] = ()
    """Run indices excluded under §11.8 — refused, or the conversation failed."""

    payloads: tuple[Any, ...] = ()
    """Parsed responses, in the same order as ``texts``.

    Text is enough for every cross-run scorer that compares what was *said*.
    It is not enough for tool calling: a target that emits only a call returns
    a null `content`, so `texts` is empty exactly where the choice being
    compared lives. Empty on a resumed run -- the bytes are in the blob store
    but nothing re-parses them yet -- which the scorer reports rather than
    reading as "chose nothing".
    """

    blob_ids: tuple[str, ...] = ()
    """Blob addresses of ``texts``, in the same order.

    Per-run observations get theirs from ``ScoreContext.text_blob_id``, but a
    cross-run scorer is handed N texts and the context carries none, so every
    determinism-family row ever written had ``blob_ids: []``. That is the one
    family whose verdict cannot then be re-checked offline against the
    responses it came from — the §5.1 guarantee the security column relied on
    when three of its verdicts turned out to be wrong.
    """


@runtime_checkable
class Scorer(Protocol):
    """One scorer family."""

    family: str
    version: int
    requires: frozenset[Capability]

    def metrics(self) -> list[MetricSpec]: ...

    def score(
        self, unit: Unit, calls: Sequence[Call], context: ScoreContext
    ) -> list[Observation]:
        """Return one or more Observations for this unit run."""
        ...


@runtime_checkable
class CrossRunScorer(Protocol):
    """A scorer that cannot work one run at a time.

    Determinism is the case that forces this: repeatability, semantic
    stability and invariance are all statements *about the set of runs*, and a
    per-run ``score()`` has nothing to compare against. Rather than let such a
    scorer smuggle state between calls, the runner hands it every run at once.
    """

    family: str
    version: int

    def finalize(
        self, evidence: Sequence[RunEvidence], context: ScoreContext
    ) -> list[Observation]:
        """Return Observations computed across all runs of all units."""
        ...


class ScorerRegistry:
    def __init__(self) -> None:
        self._scorers: dict[str, Scorer] = {}
        self._loaded_plugins = False
        self.plugin_errors: list[tuple[str, str]] = []
        """Plugins that failed to load, as (name, reason). Reported, never
        swallowed: an entry point that raises on import is exactly the case
        a user needs told about."""

    def register(self, scorer: Scorer) -> Scorer:
        if scorer.family in self._scorers:
            raise ValueError(f"scorer family {scorer.family!r} is already registered")
        self._scorers[scorer.family] = scorer
        return scorer

    def get(self, family: str) -> Scorer:
        return self._scorers[family]

    def all(self) -> tuple[Scorer, ...]:
        return tuple(self._scorers[k] for k in sorted(self._scorers))

    def applicable(self, capabilities: CapabilityReport) -> tuple[Scorer, ...]:
        """Scorers whose required capabilities are all supported."""
        return tuple(
            s
            for s in self.all()
            if all(capabilities.supports(c) for c in s.requires)
        )

    def skipped(
        self, capabilities: CapabilityReport
    ) -> tuple[tuple[Scorer, str], ...]:
        """Scorers that cannot run, each with the reason (I5).

        The reason names the capability detector that ruled it out, so a scorer
        never silently vanishes from a report.
        """
        out: list[tuple[Scorer, str]] = []
        for scorer in self.all():
            missing = [c for c in sorted(scorer.requires, key=lambda c: c.value)
                       if not capabilities.supports(c)]
            if missing:
                out.append(
                    (scorer, "; ".join(capabilities.skip_reason(c) for c in missing))
                )
        return tuple(out)

    def from_entry_points(self) -> None:
        """Load third-party scorers declared under ``sweepeval.scorers``.

        Idempotent, and called for you: :func:`registry` runs it once on first
        use. It had **no callers at all** -- not in the package, not in the
        tests -- so an installed plugin was never loaded, and I10's claim that
        "adding a scorer touches no runner code" held only because adding one
        did nothing. A unit from an unregistered family was then dropped by
        the runner with no row of any kind.

        A plugin that fails to import is reported and skipped rather than
        killing the run: a broken third-party package should not make the
        tool unusable, and silence would repeat the defect this fixes.
        """
        if self._loaded_plugins:
            return
        self._loaded_plugins = True
        for entry in entry_points(group=SCORER_ENTRY_POINT_GROUP):
            try:
                loaded = entry.load()
                scorer = (
                    loaded()
                    if callable(loaded) and not hasattr(loaded, "family")
                    else loaded
                )
                self.register(scorer)
            except Exception as error:
                # A plugin is arbitrary third-party code; a broken one must
                # not make the tool unusable.
                self.plugin_errors.append(
                    (entry.name, f"{type(error).__name__}: {error}")
                )


_REGISTRY = ScorerRegistry()


def registry() -> ScorerRegistry:
    """The scorer registry, with third-party plugins loaded."""
    _REGISTRY.from_entry_points()
    return _REGISTRY


def register(scorer: Scorer) -> Scorer:
    return _REGISTRY.register(scorer)
