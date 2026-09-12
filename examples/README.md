# Example runs

## `runs/demo-quick/` — a run against the **simulated** endpoint

This is a real, complete run of the whole pipeline: discovery, capability
detection, the sampling-effect test, a six-configuration sweep, scoring, the
paired bootstrap, and the frontier. Every number in it was produced by the
code in this repository.

**The target was a mock.** `sweepeval demo` runs against a scripted endpoint
bundled inside the package — see
[`src/sweepeval/mock/scenarios/demo.yaml`](../src/sweepeval/mock/scenarios/demo.yaml).
The numbers describe that mock's scripted behaviour and say nothing about any
real model. It is committed as a **fixture and a shape reference**, not as
evidence.

Reproduce the report offline, at zero cost, with the network off:

```bash
sweepeval report examples/runs/demo-quick --format md
sweepeval report examples/runs/demo-quick --format html --prefer security,cost
```

Both work with no key and no endpoint, which is the point: a stored run
carries its cluster tables, so it can be re-reported **and re-ranked** without
touching the target.

### What was trimmed

`calls.jsonl`, `observations.jsonl`, `blobs/` and `state/` were removed before
committing — 2.7 MB of raw log for a fixture whose purpose is the report. What
remains is what `report` and `compare` actually read:

| File | Why it is here |
|---|---|
| `manifest.json` | Comparability keys, capabilities, seeds. `compare` reads only this. |
| `plan.json` | The frozen configs, axes, shrink steps, units and canary table. |
| `aggregates.json` | Intervals and cluster tables. `report` reads only this. |
| `frontier.json` | Every comparison with its p-value and Holm threshold. |
| `report.md` | The rendered output, so the file can be read without running anything. |

A run you produce yourself keeps all of it.

## A run against a real endpoint

Not committed, and deliberately so. Publishing one means publishing a specific
provider's measured security and guardrail behaviour on a specific date, from
a single run, at a profile chosen for a README. That is a claim this project
is not in a position to stand behind, and a stale one would be worse than
none.

Produce your own:

```bash
sweepeval sweep https://your-endpoint --key $KEY --profile standard --yes
sweepeval report .sweepeval/runs/<run-id> --format html
```
