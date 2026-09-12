"""Property: comparability refusal is symmetric and reflexive-safe (spec §6.5, I6).

Asymmetry would be a real bug rather than an aesthetic one: ``compare`` and
``gate`` call this from opposite directions, so a rule that refused one way and
allowed the other would make the gate's verdict depend on argument order.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from sweepeval.schema.comparability import (
    Comparability,
    HardKeys,
    SoftKeys,
    compare_keys,
)

_profiles = st.sampled_from(["quick", "standard", "deep"])
_hashes = st.sampled_from(["a" * 64, "b" * 64])
_runs = st.integers(1, 5)
_locals = st.booleans()


def _c(profile: str, corpus: str, n_runs: int, local: bool = False) -> Comparability:
    return Comparability(
        hard=HardKeys(
            schema_major=1,
            suite_version=1,
            corpus_hash=corpus,
            probe_layers=("generic",),
            target_type="BARE_MODEL",
            similarity_backend="lexical",
            judge=None,
            profile=profile,
            pricing_source="none",
            scorer_versions={"security": 1},
            extraction_path="$.x",
        ),
        soft=SoftKeys(n_runs=n_runs, concurrency=2, tool_version="0.1.0"),
        local=local,
    )


@given(_profiles, _hashes, _runs, _locals, _profiles, _hashes, _runs, _locals)
def test_refusal_is_symmetric(
    p1: str,
    h1: str,
    n1: int,
    l1: bool,
    p2: str,
    h2: str,
    n2: int,
    l2: bool,
) -> None:
    a, b = _c(p1, h1, n1, l1), _c(p2, h2, n2, l2)
    assert compare_keys(a, b).ok == compare_keys(b, a).ok


@given(_profiles, _hashes, _runs, _locals, _profiles, _hashes, _runs, _locals)
def test_the_same_keys_are_refused_in_both_directions(
    p1: str,
    h1: str,
    n1: int,
    l1: bool,
    p2: str,
    h2: str,
    n2: int,
    l2: bool,
) -> None:
    a, b = _c(p1, h1, n1, l1), _c(p2, h2, n2, l2)
    forward = {m.key for m in compare_keys(a, b).refusals}
    backward = {m.key for m in compare_keys(b, a).refusals}
    assert forward == backward


@given(_profiles, _hashes, _runs, _locals)
def test_a_result_is_always_comparable_with_itself(
    p: str, h: str, n: int, local: bool
) -> None:
    """Including LOCAL results: a project comparing against its own baseline is
    the within-project trend §6.5 permits."""
    assert compare_keys(_c(p, h, n, local), _c(p, h, n, local)).ok
