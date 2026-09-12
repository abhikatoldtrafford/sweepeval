# Comparability

Two runs are comparable only when the things that determine *what a metric
means* are identical. sweepeval records those things as **comparability keys**
in every run's `manifest.json`, and refuses comparisons across a mismatch.

The refusal is never "results incomparable". It names the key and both values,
because a user told two runs cannot be compared needs to know which of eleven
things changed — and usually it is one they can undo.

```
refusing to compare 20260912T1030-a1 with 20260913T0900-b2 -- these runs do
not measure the same thing:
  profile differs: quick vs standard -- the profile determines the probe set
  and the retention depth weighting, so the metrics measure different things
```

## The eleven hard keys

A mismatch on any of these refuses.

| Key | Why it changes meaning |
|---|---|
| `schema_major` | The result format is incompatible. |
| `suite_version` | The shipped probe suite changed. |
| `corpus_hash` | The probes themselves changed. |
| `probe_layers` | One run included probes the other did not. |
| `target_type` | A bare model and an agent system are not comparable. |
| `similarity_backend` | Lexical and embedding similarity are different measurements. |
| `judge` | Enabling, disabling or changing the judge changes how ambiguous responses were scored. |
| `profile` | Determines the probe set *and* the retention depth weighting. |
| `pricing_source` | With no pricing the cost objective is tokens per probe — a different quantity in different units. |
| `scorer_versions` | A scorer change alters what its rate counts. |
| `extraction_path` | A different response path changes every text-derived metric. |

## The three soft keys

These warn and annotate rather than refuse: `n_runs`, `concurrency`,
`tool_version`.

## Locality

A run whose probe layers include user-supplied or generated probes is
**LOCAL**. Two local runs of the same project are comparable — that is a
within-project trend — but carry a warning, because a local result can never
be presented as cross-user comparable. A local/non-local mismatch refuses.

## Checking two runs

```bash
sweepeval compare .sweepeval/runs/<a> .sweepeval/runs/<b>
```

Pure and offline. Exit `0` comparable, `1` refused. It reads two manifests and
sends nothing, so it is safe to point at a colleague's committed run.

## Where this bites

`--resume` verifies the plan hash, the corpus hash and every hard key before
appending a single row. The failure it exists to prevent is quiet: you edit
your config, resume, and the finished run mixes rows measured under two
different definitions with nothing in the artifact saying so.
