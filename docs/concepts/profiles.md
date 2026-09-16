# Profiles, cost and wall-clock

Every run prints its estimate and asks before spending. The estimate covers
**discovery and capability detection too** — those spend first, and gating
after them would be gating after the money was gone. That is invariant I9.

| Profile | Units | Calls/run | Configs | Requests at 3 runs | Wall-clock, serial |
|---|---|---|---|---|---|
| `quick` | 40 | 60 | 6 | 1,165 | ~49 min |
| `standard` | 134 | 242 | 12 | 8,797 | ~367 min |
| `deep` | 134 | 242 | 12 | more | adds the context-ceiling search |

Requests go out **one at a time**. The column above used to be computed at a
concurrency of 2 that the executor never dispatched, so it promised half the
wait. It is still a floor rather than a forecast: it assumes 2.5s per request,
and a reasoning model is several times that. A `standard` sweep against
reasoning models measured 10.8s per request, which turns the 6,853-request row
above into roughly **23 hours** rather than five.

`standard` and `deep` also carry twelve `tool_integrity` probes and twelve
`retrieval` probes, each of which
sends the three offered tool schemas alongside the prompt -- a fixed input-token
cost per probe that the request count does not show.

`standard` and `deep` gained thirty units when the degradation family landed:
ten long-input probes, ten dispatched under load, and ten serial controls the
load metric is scored against. Ten of them carry up to 24,000 characters of
generated filler, so the family costs far more in *input tokens* than its
thirty calls suggest -- roughly 98k characters, about 25k tokens, per run per
config. The request column does not show that; the token estimate in the
pre-flight does. `quick` is unchanged: the family is not measured there.

The totals include discovery and capability detection, which is why they
exceed `units x calls x runs x configs`.

## quick is not gate-eligible

`quick` buys its speed by halving the sweep to 6 configurations, not by
cutting probes — cluster counts depend on probes, not configurations, so
cutting probes would drop families below the bootstrap floor.

Its intervals are valid but wide, so few pairs separate. Every report at
`quick` says so, and says it is not gate-eligible. Use `standard` for
decisions.

## Capping spend

```bash
sweepeval sweep https://your-endpoint --key $KEY --max-requests 2000
```

Two different things happen depending on where the cap sits:

- **Below the estimate** — a deliberate request for a partial sweep. The run
  stops *between* configurations (never mid-configuration, which would leave a
  configuration with partial coverage that can still be compared against, and
  the comparison would be wrong), reports `INCOMPLETE`, and names every
  configuration that never ran.
- **Below what discovery and capability detection need** — nothing is sent at
  all. Spending the budget on discovery and having nothing left to score with
  buys no sweep.

## The shrink ladder

The planner enumerates configurations from what discovery *proved* is
variable, then shrinks the cross product to the profile's cap by a fixed,
disclosed ladder. Every step it takes is printed and written to `plan.json`:

```
shrink ladder (cap 6)
  drop temperature=0.7 (24 -> 16 configs)
  drop system_prompt=terse_permissive (16 -> 12 configs)
  cap models (12 -> 6 configs)
```

The order is fixed — drop `top_p`, then `temperature=0.7`, then
`system_prompt=terse_permissive`, then cap models, then truncate — so the same
target and profile yield the same sweep every time. The last step exists
because the axis-level steps cannot always reach an arbitrary cap, and the cap
has to bind: the estimate you consented to was computed from it.

## Cost, and why no prices ship

No price table exists anywhere in this repo. Prices change weekly, a stale
table produces confidently wrong dollar figures, and a wrong number carries
more authority than no number.

Supply your own pricing and you get money per probe. Supply none and the cost
objective **degrades to output tokens per probe** — a different quantity in
different units, which is why `pricing_source` is a hard comparability key.
The report says which one you are looking at:

```
cost
  cfg-00     12 output tokens per probe (measured; no pricing supplied, so the
             cost objective is tokens, not money)
```

Token counts are `MEASURED` when the target reports usage and `ESTIMATED`
(`chars/4`, named in the manifest) when it does not. Below 95% usage coverage
the whole figure is ESTIMATED, because a figure that is part measurement and
part heuristic must not be labelled as a measurement.

## Resuming

Checkpoints are per `(config, unit, run)`, not per configuration. A
`standard` configuration is roughly 490 calls; losing all of it because a
crash landed at 99% is unacceptable on work you paid for.

```bash
sweepeval sweep https://your-endpoint --key $KEY --resume <run-id>
```

Resume verifies the plan hash, the corpus hash and every hard comparability
key before appending a row, and refuses on any mismatch.

## Caching

An endpoint with prompt caching returns the cached response for runs 2 and 3,
which falsifies determinism and deflates latency. Identical responses are not
the signal — a deterministic target returns those too. The signal is identical
responses whose repeats arrive *too fast to have been generated*, and it sets
`CACHE_SUSPECTED` on the configuration and on every metric a cache would
corrupt.

Per-run canaries already differ, so security probes bust naive caches by
construction; the flag catches the rest.
