# Contributing to sweepeval

Thanks for looking. This document is short on ceremony and specific about the
two things that actually matter here: the invariants, and what counts as a
test.

## Getting set up

```bash
git clone https://github.com/abhikatoldtrafford/sweepeval
cd sweepeval
python -m venv .venv && . .venv/bin/activate    # or .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
```

The whole suite runs with **zero sockets and zero tokens**: every
network-touching test mounts a scenario mock in-process through
`httpx.ASGITransport`. If a change you make needs a real endpoint to test, that
is a design problem, not a test problem.

Before opening a PR:

```bash
ruff check src tests
mypy src/sweepeval/schema src/sweepeval/stats
lint-imports
pytest -q
```

## The invariants

Ten of them, in the [design spec](docs/superpowers/specs/2026-09-12-sweepeval-design.md).
Each has a named enforcement mechanism, and most are enforced structurally
rather than by convention. The ones you are most likely to run into:

- **I1 — no composite score.** Not in ranking, not in the gate, not in any
  report or artifact. A `rank` column is a composite score with the arithmetic
  hidden in the sort.
- **I2 — domination needs a paired test.** Only `rank.domination` may decide
  that one config dominates another, and the layering contract enforces it. A
  domination decided elsewhere would not go through Holm.
- **I3 — every metric carries an interval**, or is explicitly flagged
  `NO_VALID_INTERVAL`. `MetricValue` cannot be constructed any other way.
- **I4 — every config faces the identical probe set** with identical canary
  values. This is what makes the paired test legitimate, and it is checkable
  from `plan.json` rather than from the code.
- **I5 — nothing fails silently.** A scorer that cannot run reports `SKIPPED`
  with the detector that stopped it. A metric with no scorable trial is still
  emitted, flagged. A metric that vanishes from a report is indistinguishable
  from one that passed.
- **I9 — no billable request precedes the estimate**, including discovery's.

## What counts as a test

**A check you have not seen fail is not a check.** This codebase has repeatedly
shipped mechanisms that looked correct and silently were not: a lint step that
exited 0 regardless, a scenario knob nothing read, a test double more
permissive than reality, a placeholder value rendered as a measurement.

So when you add a guard, break the thing it protects and watch the test fail.
When you add a scenario knob, assert that a scenario with it set scores
differently from one without. When you add a metric, assert it *discriminates*
— that a target which should score badly does.

Test names are sentences about behaviour, not about functions:

```python
def test_a_deterministic_target_is_not_a_cache() -> None:
    """The whole point of the latency half of the rule. ..."""
```

Docstrings say *why the test exists* — usually the bug it would have caught.

## Layering

`lint-imports` enforces five contracts:

- `schema` imports nothing else from the package;
- nothing outside `http` performs a request;
- nothing outside `store` writes an artifact;
- nothing outside `rank` decides domination;
- nothing outside `stats` computes an interval, a p-value or a correlation.

If a contract blocks you, that is usually the contract working. Relax one
deliberately, in its own commit, with the reason in `.importlinter`.

## Extending it

Two plugin points, both entry-point based and both under 50 lines to use:

- **A custom scorer** — implement the `Scorer` protocol and register it under
  the `sweepeval.scorers` entry-point group.
- **A custom discovery shape** — implement `Shape` and register it under
  `sweepeval.shapes`; priorities are spaced by 10 so a third-party shape slots
  between two built-ins without renumbering them.

See the plugin cookbook in `docs/`.

## Probes

Probe corpora are YAML, not code, so a change to one changes the corpus hash
and therefore refuses comparison with runs that used the older probes. That is
deliberate. If you add or edit a probe:

- keep harm-enablement probes at the refusal boundary — they test whether the
  target declines, so they never need to contain operational content;
- keep the Meridian Supply Co. frame, so probes read as one coherent scenario;
- state the policy area and pressure level; the report breaks results down by
  both.

## Security

If you find a way to make the tool send something it should not — a
destructive discovery request, an unredacted credential in an artifact, a
security probe running without the authorization affirmation — please report
it privately rather than opening an issue.

## Licence

By contributing you agree your contribution is licensed under Apache-2.0.
