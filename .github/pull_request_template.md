## What this changes

<!-- One or two sentences. What behaviour is different after this PR. -->

## Why

<!-- The problem. If this fixes a bug, describe what the bug produced -- a
     wrong number, a silent pass, a crash -- not just where it was. -->

## How it was verified

<!-- Required. "A check you have not seen fail is not a check."

     If this adds or changes a guard, say what you broke to watch the test
     fail. If it adds a metric or a scenario knob, say what proves it
     discriminates -- that a target which should score badly does. -->

- [ ] `ruff check src tests`
- [ ] `mypy src/sweepeval/schema src/sweepeval/stats`
- [ ] `lint-imports`
- [ ] `pytest -q`

## Invariants

<!-- Tick only what applies; delete the rest. -->

- [ ] Touches an invariant (I1-I10). Which, and how it still holds:
- [ ] Changes what a metric means, so it changes a hard comparability key
      (runs from before and after will refuse to compare -- that is correct).
- [ ] Changes a disclosed heuristic (tier 2), and CHANGELOG.md says so.
- [ ] Changes the public API or an artifact schema (tier 1), additively.
