# Quickstart

## 1. Run the demo — no endpoint, no key, no spend

```bash
uvx sweepeval demo
```

This runs the whole pipeline against a simulated endpoint bundled in the
package. Read the output top to bottom: it is the same report a real run
produces.

## 2. Look at your endpoint before spending anything

```bash
sweepeval discover https://your-endpoint --key $KEY
```

Discovery sends **at most 25 inert, read-shaped requests** under a wall-clock
cap. It prints the request shape it found, the response path it will extract
from, the confidence it has in each inference, and an editable config file you
can correct if it guessed wrong.

If it fails, it prints every shape it tried, every mutation, and every status
and error body — the transcript *is* the deliverable when discovery fails.

## 3. Score one configuration

```bash
sweepeval evaluate https://your-endpoint --key $KEY
```

## 4. Sweep

```bash
sweepeval sweep https://your-endpoint --key $KEY --profile standard --yes
```

You will be shown an estimate — requests, tokens, wall-clock — and asked
before anything is sent. `--yes` accepts it non-interactively. Without a TTY
and without `--yes`, the run **declines** rather than assuming consent.

## 5. Answer the question

The frontier can legitimately end in "everything is statistically tied". That
is an answer, not a failure. To get one config named, tell it your priority:

```bash
sweepeval sweep https://your-endpoint --key $KEY \
  --prefer "maximize security_pass_rate subject to latency_p95_ms < 2000"
```

or lexicographically:

```bash
sweepeval sweep https://your-endpoint --key $KEY --prefer security,cost,latency
```

The preference is yours, applied at report time, and changing it needs no
re-run:

```bash
sweepeval report .sweepeval/runs/<run-id> --prefer cost,security
```

## 6. Gate on it

```bash
sweepeval baseline .sweepeval/runs/<run-id> --out .sweepeval/baseline.json
git add .sweepeval/baseline.json      # designed to be committed; carries no credentials
sweepeval gate https://your-endpoint --key $KEY --baseline .sweepeval/baseline.json
```

Exit codes: `0` unchanged, `1` regressed, `2` incomparable, `3` inconclusive.
