# Declared sweeps

Zero-config is the default, not the only mode. When discovery guesses wrong,
or when you want axes it cannot see, write the config down.

## Start from what discovery found

```bash
sweepeval discover https://your-endpoint --key $KEY --out sweepeval.yaml
```

That writes an annotated config: the shape it identified, the path it will
extract from, the auth style, and its confidence in each. Every low-confidence
inference is marked, because discovery is a bootstrap you can correct rather
than an oracle you must accept.

## Correct an inference

The single most common correction is the extraction path. If the report says

```
assumptions you can correct
  extraction.text_path (low) — $.choices[0].message.content (via blind_walk)
```

then the nonce oracle did not fire — usually because the target would not
comply with `Reply with exactly ...` — and the path came from a heuristic walk.
Fix it:

```yaml
extraction:
  text_path: "$.result.completion.text"
```

Everything text-derived depends on that path, which is why it is a hard
comparability key: change it and old runs correctly refuse to compare.

## Declare axes discovery cannot see

Discovery can only sweep what it can prove is variable. A routing header, a
retrieval mode, a feature flag in your own service — none of those are
visible from outside.

```yaml
target:
  url: https://your-endpoint/v1/chat/completions
  auth: bearer

axes:
  model: [gpt-4o-mini, gpt-4o]
  temperature: [0.0, 1.0]
  headers.x-retrieval-mode: [off, hybrid, dense]

profile: standard
runs: 3
```

A declared axis is swept whether or not the sampling-effect test would have
found it effective — you asserted it matters, and the tool takes your word.
The report says which axes were declared and which were discovered.

## Pricing

Supply your own and the cost objective is money rather than tokens:

```yaml
pricing:
  input_per_mtok: 0.15
  output_per_mtok: 0.60
  currency: USD
  source: "our provider invoice, 2026-09"
```

`source` is recorded in the manifest. No price table ships with the tool, so
this is the only way to get a dollar figure — see
[profiles and cost](../concepts/profiles.md).

## Constraints and objectives

```yaml
constraints:
  - metric: error_rate
    limit: 0.02
  - metric: latency_p95_ms
    limit: 3000

objectives: [security_pass_rate, guardrail_pass_rate, latency_p95_ms]
```

Constraints are applied to the interval's **favourable bound**, not the point
estimate — a point threshold on a noisy rate is a coin flip dressed as a rule.

Narrowing objectives needs no re-run: every configuration runs the full
profile, so you can also do it at report time with `--objectives`.

## Run it

```bash
sweepeval sweep --config sweepeval.yaml --yes
```

A declared run stores its config's hash in the manifest, so `--resume` refuses
if you edit the file mid-run rather than mixing rows measured under two
different definitions.
