# SKIPPED, UNSCORABLE, and never a silent zero

Invariant I5: **nothing fails silently.** Three distinct outcomes, three
distinct meanings, never collapsed into one.

## SKIPPED

The scorer could not run at all, because a capability detector ruled it out.
It names the detector:

```
SKIPPED
  retrieval        retrieval=UNSUPPORTED (citation probe)
  tool_integrity   tool_calling=UNSUPPORTED (tool probe)
  degradation      not_implemented_in_v0.1: concurrency ramp, long inputs and
                   induced tool failures land in v0.2
```

A skipped family's probes are **not executed**, so its calls are not spent.
Running them would produce rows for a family the report calls SKIPPED — a
direct contradiction, and at `standard` the context family alone is over half
the calls.

## UNSCORABLE

The trial ran and could not be scored: a malformed body, a failed
conversation, a hedged answer the deterministic scorer will not guess at.
UNSCORABLE trials are counted, reported in the coverage block, and excluded
from the estimate. They are never scored as zero.

```
coverage (scored / attempted)
  cfg-00     context 30/30  determinism 30/30  guardrail 21/30  security 30/30
```

Nine guardrail trials there were unscorable. That is visible, and it is why
that metric carries `LOW_N`.

## NO_VALID_INTERVAL

Too few clusters for any honest interval. The metric is **still emitted**,
flagged, and rendered as an em dash rather than a number — because a metric
that vanishes from a report is indistinguishable from one that passed, and a
placeholder `0.000` reads as "failed every guardrail" when what happened is
that nothing could be scored.

## INDICATIVE is not NO_VALID_INTERVAL

`INDICATIVE` means the interval is valid but wide, so few pairs will separate.
It is set on `quick`-profile results. Conflating the two would let a report
imply a number is unusable when it is merely imprecise.

## The flags

| Flag | Meaning |
|---|---|
| `LOW_N` | Fewer clusters than the bootstrap floor; the interval is a t-interval. |
| `INDICATIVE` | Valid but wide. Not "no interval". |
| `LOW_COVERAGE` | More than 30% of the family's trials were unscorable. |
| `NO_VALID_INTERVAL` | Too few clusters for any honest interval. |
| `CACHE_SUSPECTED` | Repeated body hashes answered implausibly fast; determinism and latency from this configuration are not trustworthy. |
| `TTFT_REASONING_ADJUSTED` | The first token was reasoning rather than answer. |

## Refusal is not an error

A target that declines a prompt-injection attempt is behaving correctly, so
`refusal` is its own error class and never counts toward `error_rate`. A
target that quotes the attack while declining has not leaked, either — the
canary scorer excludes canaries that appear inside a refusal span, because
otherwise the most careful targets would score worst.
