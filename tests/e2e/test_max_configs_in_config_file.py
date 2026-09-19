"""`max_configs` belongs in the file, not only on the command line.

`profile` and `runs` were file-settable and the config cap was not, though it
moves the bill further than either. A file declaring one axis of four models
still planned twelve configurations: the planner crosses the axes discovery
proved variable against the one the file declared, up to the profile cap.

The cost of the gap, measured: a four-model sweep whose config file said in a
comment that model was the only axis estimated 9,457 requests and 394 minutes
instead of 3,169 and 132, because the flag that makes the comment true was
omitted.
"""

from __future__ import annotations

from sweepeval.execute.declared import load_declared


def write(tmp_path, body: str):
    path = tmp_path / "sweepeval.yaml"
    path.write_text(body, encoding="utf-8")
    return load_declared(path)


def test_the_file_can_cap_the_configs(tmp_path) -> None:
    declared = write(tmp_path, """
target:
  url: https://example.test/v1/chat/completions
axes:
  model: [a, b, c, d]
max_configs: 4
""")
    assert declared.max_configs == 4
    assert declared.warnings == []


def test_an_absent_cap_stays_none_so_the_profile_default_applies(tmp_path) -> None:
    declared = write(tmp_path, """
target:
  url: https://example.test/v1/chat/completions
axes:
  model: [a, b]
""")
    assert declared.max_configs is None


def test_a_nonsense_cap_warns_rather_than_silently_capping(tmp_path) -> None:
    declared = write(tmp_path, """
target:
  url: https://example.test/v1/chat/completions
max_configs: 0
""")
    assert declared.max_configs is None
    assert any("positive integer" in w for w in declared.warnings)


def test_a_non_integer_cap_warns(tmp_path) -> None:
    declared = write(tmp_path, """
target:
  url: https://example.test/v1/chat/completions
max_configs: "lots"
""")
    assert declared.max_configs is None
    assert any("positive integer" in w for w in declared.warnings)
