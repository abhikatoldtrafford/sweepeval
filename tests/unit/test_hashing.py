"""Canonical hashing (spec §6.4, §6.5, §10.3, §12.5).

Every identity in the system is a hash: ``unit_id``, ``params_hash``,
``corpus_hash``, the plan hash resume verifies against. If canonicalisation
drifts on key order, float spelling or unicode, ``unit_id`` changes between
runs and I4 stops being checkable.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from sweepeval.schema.hashing import (
    canonical_json,
    corpus_hash,
    hash_obj,
    param_hash,
    sha256_hex,
)

# --- canonical_json -------------------------------------------------------


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_is_key_order_independent_when_nested() -> None:
    left = {"outer": {"b": 1, "a": [{"z": 0, "y": 1}]}}
    right = {"outer": {"a": [{"y": 1, "z": 0}], "b": 1}}
    assert canonical_json(left) == canonical_json(right)


def test_canonical_json_is_deterministic_across_float_spelling() -> None:
    assert canonical_json({"t": 1.0}) == canonical_json({"t": 1.00})


def test_canonical_json_distinguishes_int_from_float() -> None:
    """0 and 0.0 are different sampling parameters and must not collide."""
    assert canonical_json({"t": 0}) != canonical_json({"t": 0.0})


def test_canonical_json_rejects_nan_and_infinity() -> None:
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_json({"t": bad})


def test_canonical_json_rejects_nonfinite_when_deeply_nested() -> None:
    with pytest.raises(ValueError):
        canonical_json({"a": [{"b": [1, 2, float("nan")]}]})


def test_canonical_json_rejects_nonfinite_in_a_tuple() -> None:
    with pytest.raises(ValueError):
        canonical_json({"a": (1, float("inf"))})


def test_canonical_json_is_utf8_not_escaped() -> None:
    assert "é".encode() in canonical_json({"k": "é"})


def test_canonical_json_has_no_incidental_whitespace() -> None:
    assert canonical_json({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'


def test_canonical_json_rejects_non_string_keys() -> None:
    """json would coerce 1 and "1" to the same key, silently colliding."""
    with pytest.raises((TypeError, ValueError)):
        canonical_json({1: "a", "1": "b"})


# --- hash_obj / sha256_hex ------------------------------------------------


def test_sha256_hex_is_64_lowercase_hex() -> None:
    digest = sha256_hex(b"x")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_hash_obj_is_stable() -> None:
    assert hash_obj({"a": [1, 2, {"c": None}]}) == hash_obj({"a": [1, 2, {"c": None}]})


def test_hash_obj_is_stable_across_processes() -> None:
    """PYTHONHASHSEED must not reach the digest.

    Dict iteration order is seed-dependent; a canonicaliser that leaked it
    would produce a corpus_hash that changes between CI runs and refuse every
    comparison for no reason.
    """
    code = (
        "from sweepeval.schema.hashing import hash_obj;"
        "print(hash_obj({'b': 1, 'a': 2, 'c': [3, {'e': 4, 'd': 5}]}))"
    )
    digests = set()
    for seed in (0, 1, 12345):
        env = {**os.environ, "PYTHONHASHSEED": str(seed)}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1, digests


# --- param_hash -----------------------------------------------------------


def test_param_hash_is_16_hex_chars() -> None:
    digest = param_hash({"temperature": 0.0, "model": "m"})
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


def test_param_hash_distinguishes_values() -> None:
    assert param_hash({"temperature": 0.0}) != param_hash({"temperature": 0.7})


def test_param_hash_is_key_order_independent() -> None:
    assert param_hash({"a": 1, "b": 2}) == param_hash({"b": 2, "a": 1})


def test_param_hash_distinguishes_absent_from_null() -> None:
    """An unset parameter and one explicitly sent as null are different requests."""
    assert param_hash({"a": 1}) != param_hash({"a": 1, "top_p": None})


# --- corpus_hash ----------------------------------------------------------


def test_corpus_hash_changes_when_any_template_changes() -> None:
    assert corpus_hash([b"x", b"y"], 1, {"quick": ["x"]}) != corpus_hash(
        [b"x", b"z"], 1, {"quick": ["x"]}
    )


def test_corpus_hash_is_order_independent_over_templates() -> None:
    assert corpus_hash([b"x", b"y"], 1, {}) == corpus_hash([b"y", b"x"], 1, {})


def test_corpus_hash_changes_when_suite_version_changes() -> None:
    assert corpus_hash([b"x"], 1, {}) != corpus_hash([b"x"], 2, {})


def test_corpus_hash_changes_when_profile_definition_changes() -> None:
    assert corpus_hash([b"x"], 1, {"quick": ["x"]}) != corpus_hash(
        [b"x"], 1, {"quick": ["x", "y"]}
    )


def test_corpus_hash_distinguishes_a_split_template() -> None:
    """Concatenation must not alias.

    Hashing each template before folding means [b"ab"] and [b"a", b"b"] differ;
    a naive digest.update(blob) loop would collide them, so a corpus edit that
    split one probe into two would go undetected.
    """
    assert corpus_hash([b"ab"], 1, {}) != corpus_hash([b"a", b"b"], 1, {})


def test_corpus_hash_is_64_hex_chars() -> None:
    assert len(corpus_hash([b"x"], 1, {})) == 64


def test_corpus_hash_rejects_nonfinite_in_profile_defs() -> None:
    with pytest.raises(ValueError):
        corpus_hash([b"x"], 1, {"quick": {"weight": float("nan")}})


# --- documented format ----------------------------------------------------


def test_canonical_json_output_is_valid_json() -> None:
    payload = {"a": 1, "b": [1, 2, None], "c": {"d": "é"}}
    assert json.loads(canonical_json(payload).decode("utf-8")) == payload
