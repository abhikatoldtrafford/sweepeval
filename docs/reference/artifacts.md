# Artifacts

Everything a run produces, and what each file is for. All of it is append-only
or regenerable, and all of it is redacted.

```
.sweepeval/
  runs/<run_id>/
    manifest.json           comparability keys, versions, hashes, seeds,
                            capabilities, budget, authorization record
    plan.json               configs, axes, shrink steps, serialised units,
                            the canary table
    calls.jsonl             append-only, one row per HTTP call
    observations.jsonl      append-only, one row per
                            (config, unit, run, scorer, metric)
    aggregates.json         intervals and cluster tables; enough to re-report
                            and re-rank offline
    frontier.json           clusters, dominators, violators, every comparison
    blobs/<sha256>          content-addressed extracted text
    state/<config_id>.json  completed unit-runs, budget counters
    report.*                only when requested
  discovery/<url_hash>/     ladder transcript, evidence, emitted config
```

`run_id` is `YYYYMMDDTHHMMSSmmm-xxxxxx`: chronologically sortable, so listing
runs newest-first needs no manifest reads, and suffixed with randomness
because two runs can start in the same millisecond.

## manifest.json

What was run, against what, and under which rules. The comparability keys live
here, and `compare` and `--resume` read nothing else.

```json
{
  "run_id": "20260912T103015123-a1b2c3",
  "tool_version": "0.1.0",
  "schema_major": 1,
  "plan_hash": "9f2c…",
  "corpus_hash": "3ab1…",
  "comparability": {"hard": {...}, "soft": {...}, "local": false},
  "capabilities": {"system_prompt": {"verdict": "SUPPORTED", ...}},
  "target": {"url": "…", "shape": "openai.chat_completions", "auth": "bearer"},
  "authorization": {"host": "…", "affirmed_at": "…"},
  "seeds": {"seed": 0, "master_seed": "20260912T103015123-a1b2c3:0"},
  "heuristics": {"tokens": "chars/4", "wall_clock": "2.5s per request"}
}
```

`heuristics` is not decoration. An estimate presented as a measurement is
worse than no estimate, so every disclosed approximation is named beside the
numbers derived from it.

## plan.json

Written **before** the first scoring request and never rewritten. This is what
makes I4 checkable from the artifact rather than from the code: the serialised
units and the canary table are written **once, outside the config list**,
precisely because they are shared. Writing them per configuration would make
an I4 violation representable in the file.

```json
{
  "cap": 12,
  "axes": {"model": [...], "temperature": [0.0, 1.0]},
  "shrink_steps": ["drop temperature=0.7 (24 -> 16 configs)"],
  "rejected_axes": [["top_p", "INERT: …"]],
  "configs": [{"config_id": "cfg-00", "label": "…", "params": {...}}],
  "units": [{"unit_id": "u#…", "family": "security", "turns": [...]}],
  "canary_table": {"u#…|0|primary": "K7DQ3XZM2P"},
  "determinism_sharing": {"cfg-01": "cfg-00"}
}
```

Canary values differ across `run_idx` for the same unit, so a cached response
cannot pass by replaying an old canary, and are identical across
configurations, which is what makes the paired test legitimate.

The plan **hash** excludes `run_id`, `master_seed` and the canary table: a
resumed run reuses all three, and including them would make the hash useless
for the one comparison it is for — this plan against the plan your edited
config would produce now.

## calls.jsonl

One row per HTTP call, including retries. Operational metrics derive from this
file; every other family derives from `observations.jsonl`. A depth-15
conversation is fifteen rows here and one observation there.

`counts_toward_latency` is computed on the row: first attempt, and no error.
A call that succeeded only after 30s of backoff describes the rate limiter,
and a 500 that took 30s belongs in `error_rate`.

`timing.total_ms` **excludes** `queue_ms`, so the harness's own queueing under
concurrency is not measured as target latency.

## observations.jsonl

One row per `(config_id, unit_id, run_idx, scorer, metric)` — a five-part key.
Every row carries a `verdict`, and `UNSCORABLE` and `SKIPPED` rows carry a
non-blank `reason`, enforced by the schema.

Orphan rows from a unit-run that crashed mid-write are left in place;
aggregation reads only unit-runs the state file marks complete. That is how
partial work is excluded without ever rewriting an append-only log.

## aggregates.json

Derived and regenerable, but stored because a report has to be reproducible
from the run alone. It carries per-configuration intervals **and the cluster
tables behind them** — those are what the paired test resamples, so without
them a stored run could be re-reported but never re-ranked, and `--prefer`
and `--objectives` would quietly require the network again.

```bash
sweepeval report .sweepeval/runs/<id> --format html --objectives security,cost
```

## frontier.json

Every ordered pair, with its p-value and its Holm threshold, whether or not it
dominated. A reader who disagrees with a verdict has to be able to see the
evidence, and a reader who wonders why an obvious winner did not dominate has
to be able to find the objective that blocked it.

**No scalar rank field appears anywhere in this document**, and a test walks
it asserting so. A rank column is a composite score with the arithmetic hidden
in the sort.

## blobs/

Content-addressed by SHA-256, so identical responses dedupe for free. Stored
by default: the normalised extracted text of every call, capped at 64 KiB.
Stored **always**, regardless of the cap or `--no-store-bodies`: discovery
transcripts, error bodies, and every hard-fail hit.

## state/<config_id>.json

Completed unit-runs and budget counters, written atomically. Checkpoints are
per `(config, unit, run)`, not per configuration: a `standard` configuration
is roughly 490 calls, and losing all of it because a crash landed at 99% is
unacceptable on work you paid for.

A unit-run is marked complete only **after** its rows are durably appended, so
a crash between the two leaves orphan rows that aggregation skips rather than
a checkpoint claiming work that was never stored.

## What is safe to commit

`baseline.json` — yes. It is designed for it and carries no credentials.

`.sweepeval/` — no. `sweepeval init` writes the `.gitignore` entries and
prints exactly what is and is not safe to commit.

Every artifact is redacted: keys in headers, in query strings, in bodies, in
error excerpts. Redaction happens in the store, and the store is the only way
to obtain a writer, which is what makes the no-secrets contract test
exhaustive rather than a spot check.
