"""Comparability keys and refusal messages (spec §6.5). I6 load-bearing.

I6: "Results whose hard comparability keys differ refuse to be compared."

Refusal is never the string "results incomparable". Every refusal names the
key and both values, because a user who is told two runs cannot be compared
needs to know which of eleven things changed — and usually it is one they can
undo.

Four of the eleven hard keys were soft in rev 1 and were promoted in rev 2,
each because it changes what a metric *means*: ``profile`` redefines both the
probe set and the retention weighting; ``pricing_source`` switches
``cost_per_probe`` to ``tokens_out_per_probe``, a different quantity in
different units; a ``scorer_versions`` bump changes a rate's definition; and a
different ``extraction_path`` changes every text-derived metric.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_serializer

__all__ = [
    "Comparability",
    "ComparabilityVerdict",
    "HardKeys",
    "JudgeKey",
    "KeyMismatch",
    "SoftKeys",
    "compare_keys",
]


def _short(value: Any) -> str:
    text = str(value)
    return f"{text[:8]}…" if len(text) > 12 else text


# One template per hard key (§6.5). Adding a hard key without adding a message
# is caught by ``test_every_hard_key_has_a_refusal_message``.
_MESSAGES: dict[str, str] = {
    "schema_major": (
        "schema major version differs: {a} vs {b} — these results were written by "
        "incompatible versions of the result format"
    ),
    "suite_version": (
        "suite version differs: {a} vs {b} — the shipped probe suite changed between "
        "these runs"
    ),
    "corpus_hash": (
        "corpus hash differs: {a} vs {b} — the probe corpus changed between these runs"
    ),
    "probe_layers": (
        "probe layers differ: {a} vs {b} — one run included probes the other did not"
    ),
    "target_type": (
        "target type differs: {a} vs {b} — a bare model and an agent system are not "
        "comparable"
    ),
    "similarity_backend": (
        "similarity backend differs: {a} vs {b} — semantic stability computed "
        "lexically and by embedding are different measurements"
    ),
    "judge": (
        "judge differs: {a} vs {b} — enabling, disabling or changing the judge "
        "changes how ambiguous responses were scored"
    ),
    "profile": (
        "profile differs: {a} vs {b} — the profile determines the probe set and the "
        "retention depth weighting, so the metrics measure different things"
    ),
    "pricing_source": (
        "pricing source differs: {a} vs {b} — with no pricing the cost objective is "
        "tokens per probe, which is a different quantity in different units"
    ),
    "scorer_versions": (
        "scorer versions differ: {a} vs {b} — a scorer change alters what its rate "
        "counts, so the same number does not mean the same thing"
    ),
    "extraction_path": (
        "extraction path differs: {a} vs {b} — a different response path changes "
        "every text-derived metric"
    ),
}


class JudgeKey(BaseModel):
    """Judge identity (§11.9). Absent means deterministic scoring only."""

    model_config = ConfigDict(frozen=True)

    model: str
    prompt_version: int


class HardKeys(BaseModel):
    """Mismatch on any of these refuses the comparison."""

    model_config = ConfigDict(frozen=True)

    schema_major: int
    suite_version: int
    corpus_hash: str
    probe_layers: tuple[str, ...]
    target_type: str
    similarity_backend: str
    judge: JudgeKey | None
    profile: str
    pricing_source: str
    scorer_versions: dict[str, int]
    extraction_path: str

    @field_serializer("scorer_versions")
    def _serialise_scorer_versions(self, value: dict[str, int]) -> dict[str, int]:
        """Sorted, so the manifest hashes identically in every process."""
        return dict(sorted(value.items()))


class SoftKeys(BaseModel):
    """Mismatch on these warns and annotates, but does not refuse."""

    model_config = ConfigDict(frozen=True)

    n_runs: int
    concurrency: int
    tool_version: str


class Comparability(BaseModel):
    model_config = ConfigDict(frozen=True)

    hard: HardKeys
    soft: SoftKeys
    local: bool = False
    """True when probe layers include ``user`` or ``generated``.

    A local run is a within-project trend and can never be presented as
    cross-user comparable (§6.5).
    """


class KeyMismatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    a: str
    b: str
    message: str


class ComparabilityVerdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    refusals: tuple[KeyMismatch, ...] = ()
    warnings: tuple[KeyMismatch, ...] = ()

    def explain(self) -> str:
        if self.ok and not self.warnings:
            return "comparable"
        lines = [f"REFUSED: {m.message}" for m in self.refusals]
        lines += [f"warning: {m.message}" for m in self.warnings]
        return "\n".join(lines)


def _canonical(value: Any) -> Any:
    """Order-insensitive view of a key's value, for comparison only."""
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    if isinstance(value, BaseModel):
        return tuple(sorted(value.model_dump().items()))
    return value


def compare_keys(a: Comparability, b: Comparability) -> ComparabilityVerdict:
    """Decide whether two results may be compared (I6)."""
    refusals: list[KeyMismatch] = []
    warnings: list[KeyMismatch] = []

    for key in HardKeys.model_fields:
        left = getattr(a.hard, key)
        right = getattr(b.hard, key)
        if _canonical(left) != _canonical(right):
            refusals.append(
                KeyMismatch(
                    key=key,
                    a=str(left),
                    b=str(right),
                    message=_MESSAGES[key].format(a=_short(left), b=_short(right)),
                )
            )

    if a.local != b.local:
        # A local/non-local mismatch is a refusal: one run used user-supplied or
        # generated probes and the other did not, so they measure different
        # things. Two LOCAL runs of the same project ARE comparable — §6.5 calls
        # that a within-project trend — but they carry a warning, because a
        # local result can never be presented as cross-user comparable.
        refusals.append(
            KeyMismatch(
                key="local",
                a=str(a.local),
                b=str(b.local),
                message=(
                    "locality differs: one run is LOCAL (user-supplied or generated "
                    "probes) and the other is not, so they did not face the same "
                    "corpus"
                ),
            )
        )
    elif a.local:
        warnings.append(
            KeyMismatch(
                key="local",
                a="True",
                b="True",
                message=(
                    "both runs are LOCAL: comparable as a within-project trend, but "
                    "never presentable as cross-user comparable"
                ),
            )
        )

    for key in SoftKeys.model_fields:
        left = getattr(a.soft, key)
        right = getattr(b.soft, key)
        if left != right:
            warnings.append(
                KeyMismatch(
                    key=key,
                    a=str(left),
                    b=str(right),
                    message=f"{key} differs: {left} vs {right} — comparable, but noted",
                )
            )

    return ComparabilityVerdict(
        ok=not refusals, refusals=tuple(refusals), warnings=tuple(warnings)
    )
