# Gating in CI

A regression gate that does not flap.

## Why gates flap

A gate on point estimates asks "is today's number worse than the baseline's?"
On a stochastic target with N=3, the answer is yes about half the time when
nothing changed. Teams notice, and they disable the gate.

sweepeval gates on a **paired test against a committed baseline**, using the
same statistics as the frontier. Measured false-fire rate: **0.0033** on one
objective, **0.0133** across six — against roughly 0.083 per metric for the
interval-overlap rule an earlier design used, which compounds to ~0.39 across
six.

## Set it up

```bash
# Once, from a run you trust.
sweepeval sweep https://your-endpoint --key $KEY --profile standard --yes
sweepeval baseline .sweepeval/runs/<run-id> --out .sweepeval/baseline.json
git add .sweepeval/baseline.json
```

`baseline.json` is designed to be committed. It carries per-cluster values —
the paired test needs the matched blocks, not just the summary — and **no
credentials**. A contract test asserts that.

## Run it

```bash
sweepeval gate https://your-endpoint --key $KEY --baseline .sweepeval/baseline.json
```

| Exit | Meaning |
|---|---|
| `0` | Unchanged, or improved. |
| `1` | Regressed: at least one objective is significantly worse. |
| `2` | Incomparable: a hard comparability key differs. The message names it. |
| `3` | Inconclusive: not enough evidence either way. |

Exit `2` is worth handling separately in CI. It usually means something you
changed on purpose — the profile, the extraction path, a scorer version — and
the fix is to re-baseline, not to investigate a regression that did not happen.

## GitHub Actions

```yaml
name: eval
on: [pull_request]

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pipx install sweepeval
      - name: gate against the committed baseline
        run: |
          sweepeval gate ${{ vars.ENDPOINT }} \
            --key ${{ secrets.LLM_API_KEY }} \
            --baseline .sweepeval/baseline.json \
            --i-am-authorized
```

Under `GITHUB_ACTIONS`, annotations are emitted automatically — that is the
one case where the user cannot see the terminal.

For the full sweep as a scheduled job, with a JUnit report in the test tab:

```yaml
  sweep:
    runs-on: ubuntu-latest
    steps:
      - run: pipx install sweepeval
      - run: |
          sweepeval sweep ${{ vars.ENDPOINT }} \
            --key ${{ secrets.LLM_API_KEY }} \
            --profile standard --yes --i-am-authorized \
            --format junit,html
      - uses: actions/upload-artifact@v4
        with:
          name: sweepeval
          path: .sweepeval/runs/
```

## Three things to get right

**Use `--profile standard`.** `quick` is not gate-eligible; its intervals are
too wide for a gate to mean anything, and the report says so.

**Pass `--yes`.** Without a TTY and without it, the run declines rather than
assuming consent — deliberately. A CI job that starts spending thousands of
requests because nobody was there to say no is the failure the pre-flight
exists to prevent.

**Pass `--i-am-authorized` only if you are.** The security suite is an active
prompt-injection test. Against an agent system it can trigger real tool calls,
writes and spend. See [safety](safety.md).

## Docker

```yaml
      - run: |
          docker run --rm -v "$PWD/.sweepeval:/work/.sweepeval" \
            ghcr.io/abhikatoldtrafford/sweepeval \
            gate "$ENDPOINT" --key "$KEY" --baseline .sweepeval/baseline.json
```

The image runs as a non-root user, so the mounted directory does not come back
root-owned.
