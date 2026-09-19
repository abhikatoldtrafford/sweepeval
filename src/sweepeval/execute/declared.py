"""Declared sweeps: overriding what discovery inferred (spec §4.1, §8.6).

Zero-config is the default, not the only mode. Discovery is a **bootstrap you
can correct**, not an oracle you must accept, and the annotated config it
emits marks every low-confidence inference precisely so you can fix it.

Three kinds of override, in increasing order of how much they change:

``extraction.text_path``
    The single most common correction. Everything text-derived depends on it,
    which is why it is a hard comparability key: change it and old runs
    correctly refuse to compare.
``axes``
    Axes discovery cannot see. A routing header, a retrieval mode, a feature
    flag in your own service — none are visible from outside. A declared axis
    is swept whether or not the sampling-effect test would have called it
    effective, because you asserted it matters.
``pricing``, ``constraints``, ``objectives``, ``profile``, ``runs``,
``max_configs``
    Run parameters. ``pricing`` is the only way to get a dollar figure, since
    no price table ships (§12.4).

Everything here is **additive to** discovery, never a replacement for it. The
endpoint still has to be probed: a declared config that skipped discovery
would have no capability report, and a scorer with no capability report cannot
say why it was skipped.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sweepeval.execute.cost import Pricing
from sweepeval.rank.constraints import Constraint

__all__ = ["DeclaredConfig", "load_declared"]

_AXIS_KEYS = ("model", "temperature", "top_p", "system_prompt")


@dataclass
class DeclaredConfig:
    """What a ``sweepeval.yaml`` can say."""

    path: Path | None = None
    url: str | None = None
    key_env: str | None = None
    text_path: str | None = None
    axes: dict[str, list[Any]] = field(default_factory=dict)
    pricing: Pricing | None = None
    constraints: tuple[Constraint, ...] = ()
    objectives: tuple[str, ...] = ()
    profile: str | None = None
    runs: int | None = None
    max_configs: int | None = None
    """The config cap, in the file rather than only on the command line.

    `profile` and `runs` were file-settable and this was not, though it moves
    the bill further than either: a file declaring one axis of four models
    still planned twelve configurations, because the planner crosses the axes
    discovery proved variable against the one the file declared, up to the
    profile cap. Writing "model is the only axis" in a comment does not make
    it so, and the flag that does is easy to omit -- which cost a 3x
    over-estimate and a cancelled run before this existed.
    """
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> list[str]:
        """Lines for the report: what the user overrode, and with what."""
        lines: list[str] = []
        if self.text_path:
            lines.append(f"extraction.text_path -> {self.text_path}")
        for axis, values in sorted(self.axes.items()):
            lines.append(f"axis {axis} declared: {', '.join(str(v) for v in values)}")
        if self.pricing is not None:
            lines.append(f"pricing from {self.pricing.source}")
        for constraint in self.constraints:
            lines.append(f"constraint {constraint.describe()}")
        if self.objectives:
            lines.append(f"objectives narrowed to {', '.join(self.objectives)}")
        return lines


def load_declared(path: Path | str) -> DeclaredConfig:
    """Read a config, ignoring the annotations discovery wrote into it.

    The emitted file is full of ``*_confidence``, ``*_method`` and
    ``evidence`` keys. Those are there for the reader, and reading them back
    as settings would let a stale annotation quietly become an instruction.
    """
    from sweepeval.discovery.emit import load_config

    file = Path(path)
    payload = load_config(file)
    declared = DeclaredConfig(path=file)

    target = payload.get("target")
    if isinstance(target, Mapping):
        url = target.get("url")
        if isinstance(url, str) and url:
            declared.url = url
        key_env = target.get("key_env")
        if isinstance(key_env, str) and key_env:
            declared.key_env = key_env

    extraction = payload.get("extraction")
    if isinstance(extraction, Mapping):
        text_path = extraction.get("text_path")
        if isinstance(text_path, str) and text_path:
            declared.text_path = text_path

    declared.axes = _axes(payload.get("axes"), declared.warnings)
    declared.pricing = _pricing(payload.get("pricing"), declared.warnings)
    declared.constraints = _constraints(payload.get("constraints"), declared.warnings)

    objectives = payload.get("objectives")
    if isinstance(objectives, Sequence) and not isinstance(objectives, str):
        declared.objectives = tuple(str(o) for o in objectives)

    profile = payload.get("profile")
    if isinstance(profile, str):
        if profile in ("quick", "standard", "deep"):
            declared.profile = profile
        else:
            declared.warnings.append(
                f"unknown profile {profile!r}; using the command-line value"
            )

    runs = payload.get("runs")
    if isinstance(runs, int) and runs > 0:
        declared.runs = runs

    max_configs = payload.get("max_configs")
    if isinstance(max_configs, int) and max_configs > 0:
        declared.max_configs = max_configs
    elif max_configs is not None:
        declared.warnings.append(
            f"max_configs must be a positive integer, got {max_configs!r}; "
            f"using the profile default"
        )

    return declared


def _axes(value: Any, warnings: list[str]) -> dict[str, list[Any]]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, list[Any]] = {}
    for name, values in value.items():
        key = str(name)
        if not isinstance(values, Sequence) or isinstance(values, str):
            warnings.append(f"axis {key!r} is not a list; ignored")
            continue
        listed = list(values)
        if len(listed) < 2:
            # A one-value axis is not an axis. Silently keeping it would put a
            # column in plan.json that never varies, which reads as a swept
            # dimension that found nothing.
            warnings.append(
                f"axis {key!r} has {len(listed)} value(s); an axis needs at least 2"
            )
            continue
        if key not in _AXIS_KEYS and not key.startswith("headers."):
            warnings.append(
                f"axis {key!r} is not a sampling parameter or a headers.* axis; "
                "it will be sent as a request-body field"
            )
        out[key] = listed
    return out


def _pricing(value: Any, warnings: list[str]) -> Pricing | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return Pricing(
            input_per_mtok=float(value["input_per_mtok"]),
            output_per_mtok=float(value["output_per_mtok"]),
            currency=str(value.get("currency", "USD")),
            source=str(value.get("source", "sweepeval.yaml")),
        )
    except (KeyError, TypeError, ValueError):
        warnings.append(
            "pricing needs input_per_mtok and output_per_mtok as numbers; "
            "ignored, so the cost objective stays in tokens"
        )
        return None


def _constraints(value: Any, warnings: list[str]) -> tuple[Constraint, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    out: list[Constraint] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            warnings.append(f"constraint {entry!r} is not a mapping; ignored")
            continue
        metric = entry.get("metric")
        limit = entry.get("limit")
        if not isinstance(metric, str) or not isinstance(limit, (int, float)):
            warnings.append(
                f"constraint {entry!r} needs a metric and a numeric limit; ignored"
            )
            continue
        out.append(
            Constraint(
                metric=metric,
                limit=float(limit),
                direction=str(entry.get("direction", "max")),
                kind=str(entry.get("kind", "interval")),
            )
        )
    return tuple(out)
