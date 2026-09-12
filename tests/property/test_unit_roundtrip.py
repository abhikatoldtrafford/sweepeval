"""Property: Units survive the plan.json round trip unchanged (spec §6.1, I4)."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from sweepeval.schema.unit import ScoringContract, Turn, Unit

_text = st.text(min_size=1, max_size=40)
_profiles = st.sets(st.sampled_from(["quick", "standard", "deep"]), min_size=1)


@given(
    template_id=st.from_regex(r"[a-z]{3}\.[a-z]{3}\.v1", fullmatch=True),
    turn_texts=st.lists(_text, min_size=1, max_size=5),
    params=st.dictionaries(
        st.sampled_from(["depth", "level", "variant"]),
        st.integers(0, 30),
        max_size=3,
    ),
    profiles=_profiles,
)
def test_unit_survives_json_roundtrip_unchanged(
    template_id: str,
    turn_texts: list[str],
    params: dict[str, int],
    profiles: set[str],
) -> None:
    original = Unit.make(
        template_id=template_id,
        family="security",
        turns=[Turn(role="user", text=t) for t in turn_texts],
        scoring=[ScoringContract(kind="canary_absent", canary="primary")],
        profiles=profiles,
        params=params,
        canary_names=("primary",),
    )
    restored = Unit.model_validate_json(original.model_dump_json())

    assert restored == original
    assert restored.unit_id == original.unit_id
    assert restored.calls_per_run == original.calls_per_run
    # Byte-identity, not just equality: plan.json is hashed (§12.5).
    assert restored.model_dump_json() == original.model_dump_json()


@given(
    turn_texts=st.lists(_text, min_size=1, max_size=6),
    profiles=_profiles,
)
def test_calls_per_run_always_equals_turn_count(
    turn_texts: list[str], profiles: set[str]
) -> None:
    unit = Unit.make(
        template_id="abc.def.v1",
        family="context",
        turns=[Turn(role="user", text=t) for t in turn_texts],
        scoring=[ScoringContract(kind="fact_recall")],
        profiles=profiles,
    )
    assert unit.calls_per_run == len(turn_texts)


@given(
    params_a=st.dictionaries(st.sampled_from(["depth", "level"]), st.integers(0, 5)),
    params_b=st.dictionaries(st.sampled_from(["depth", "level"]), st.integers(0, 5)),
)
def test_unit_id_agrees_with_params_equality(
    params_a: dict[str, int], params_b: dict[str, int]
) -> None:
    """Same params means same id; different params means different id."""

    def build(params: dict[str, int]) -> Unit:
        return Unit.make(
            template_id="abc.def.v1",
            family="security",
            turns=[Turn(role="user", text="x")],
            scoring=[ScoringContract(kind="canary_absent", canary="primary")],
            profiles={"standard"},
            params=params,
        )

    assert (build(params_a).unit_id == build(params_b).unit_id) == (params_a == params_b)
