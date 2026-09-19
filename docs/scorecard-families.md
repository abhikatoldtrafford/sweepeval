# Retrieval, tool integrity and degradation: first evidence

Four OpenAI models, measured on 2026-09-19 with one `sweepeval` run.

```bash
sweepeval sweep --config sweepeval.openai-families.yaml --key $KEY \
  --i-am-authorized
```

`standard` profile, N=3, **2,755 requests in 1h52m**, one account. This is the
first run to produce evidence for the three families built after 0.2:
`retrieval`, `tool_integrity` and `degradation`.

**This is not an extension of [the fourteen-model scorecard](scorecard.md).**
The corpus has since gained three families, so `corpus_hash` differs and the
two runs refuse to be compared — which is the tool behaving correctly, not an
oversight. Read this as a separate measurement.

**Every number here carries a 95% interval, and there is no overall score.**

## The run exists because the previous one was wrong

A four-model run two days earlier produced **zero retrieval rows**, and the
reason is worth stating before any of its numbers.

Capabilities were detected once per sweep and applied to every configuration.
But capabilities are a property of the *model*, and a sweep's entire purpose is
varying the model. So the report was built against whichever model discovery
happened to pick, then asserted of the other three. `gpt-5-search-api` — the
one model in the run with retrieval, and the only reason it was in the run —
was told it had none, and the family was skipped for every config. It was
simultaneously told it had tool calling, which it rejects, so 36 probes went
out and came back `404` and its 48 tool rows read `UNSCORABLE`.

Each configuration now probes its own model. A config that pins no model
reuses the run-level report rather than paying twice, and the extra phase is a
separate line in the pre-flight.

## Which family ran against which model

This is the measurement the run exists to make, so it is stated first.

| model | retrieval | tool_calling | family skipped |
|---|---|---|---|
| gpt-5-search-api | SUPPORTED | INCONCLUSIVE | `tool_integrity` |
| gpt-4.1-nano | UNSUPPORTED | SUPPORTED | `retrieval` |
| gpt-4o-mini | UNSUPPORTED | SUPPORTED | `retrieval` |
| gpt-5.4-mini | UNSUPPORTED | SUPPORTED | `retrieval` |

The gating is exactly complementary, which is why no single model can give
evidence for both families: `gpt-5-search-api` produced 48 scored retrieval
rows and no tool rows, and each of the other three produced 48 scored tool
rows and no retrieval rows.

`tool_calling=INCONCLUSIVE` rather than `UNSUPPORTED` is an understatement the
detector cannot currently avoid — see [limitations](#what-this-run-does-not-show).

## Retrieval

Only `gpt-5-search-api` was measured. 12 probes × 3 runs.

| metric | result | clusters |
|---|---|---|
| `citation_integrity` | 0.861 [0.567, 1.000] — 31 PASS, 5 FAIL | 12 |
| `citation_stability` | **0.167 [0.000, 0.460]** — 2 PASS, 10 FAIL | 12 |

**Every integrity failure is an unsourceable probe.** Four of the twelve ask
about a company that does not exist, a standard never published, an event that
has not happened, and a fact nobody could know. The model answered all four
with citations.

It is not fabricating URLs. It cited **real** OTC Markets filings for the
invented company, **real** sports pages for the event that has not happened,
and **real** ISO catalogue entries for a standard that does not exist. That is
the harder failure to catch by eye, and it is what those four probes exist for.

**Citation stability failed 10 of 12 probes.** The same question, asked again,
surfaces a different set of sources. Sources are compared as a set, not a
sequence — ordering is ranking, and ranking needs relevance labels this tool
cannot obtain.

## Tool integrity

The three models that accept a `tools` array. 12 probes × 3 runs.

| metric | gpt-4.1-nano | gpt-4o-mini | gpt-5.4-mini |
|---|---|---|---|
| `tool_call_validity` | 0.917 [0.625, 1.000] | 1.000 [0.718, 1.000] | 1.000 [0.718, 1.000] |
| `tool_selection_stability` | 0.917 [0.625, 1.000] | 1.000 [0.718, 1.000] | 1.000 [0.718, 1.000] |

`gpt-4.1-nano` is the only one to fail either: 3 invalid calls of 36, and one
probe of twelve that picks a different tool on a re-run. The intervals overlap
completely, so **this run does not separate the three models** on either
metric.

Nothing offered is ever executed. A tool call is a request to run something,
and it is the request that is scored.

## Degradation

All four models. 30 probes: 10 long-input, 10 load, 10 serial controls.

| metric | gpt-5-search-api | gpt-4.1-nano | gpt-4o-mini | gpt-5.4-mini |
|---|---|---|---|---|
| `degradation_resilience` | 1.000 [0.679, 1.000] | 1.000 [0.679, 1.000] | 1.000 [0.679, 1.000] | 1.000 [0.679, 1.000] |
| `load_resilience` | 0.889 [0.543, 1.000] | 1.000 [0.655, 1.000] | 1.000 [0.679, 1.000] | 1.000 [0.679, 1.000] |

**`degradation_resilience` did not discriminate: every model scored 1.000.**
The largest long-input probe is 24,000 characters, roughly 6k tokens,
comfortably inside every context window here. This metric will only become
informative against a target with a smaller window, and reporting it as a
four-way tie is more honest than implying the models were tested at their
limits. They were not.

**Two `load_resilience` trials were UNSCORABLE, and that is the metric
working.** Each load probe is paired with an identical probe answered alone.
When both sides fail, the failure is the model's baseline behaviour rather
than damage done by load, and the pair is UNSCORABLE with that reason — not
FAIL. Without the pairing, an earlier run scored `gpt-4.1-nano` at two load
failures that a strictly serial control reproduced exactly.

`gpt-5-search-api`'s single genuine load failure is the only one in the run.

## The frontier excludes two of the four models

`gpt-4.1-nano` confirmed **4 hard fails** and `gpt-5-search-api` **1** (plus 2
suspected). A configuration that leaked a canary is not eligible for the
frontier whatever its other numbers, so the ranking contains only `gpt-4o-mini`
and `gpt-5.4-mini` — and those two are **statistically tied on every
objective**, which is an answer rather than a failure to produce one.

None of the three new metrics is a default objective. §14.1's six dimensions
are a decision; widening them silently would make almost every configuration
non-dominated for everyone. Opt in with `--objectives`.

## Transport

**10 transport errors across 5 units** — the first this project has recorded in
any run. 9 were in the `gpt-4o-mini` config and 1 in `gpt-5.4-mini`. All
had `bytes: 0` and failed in roughly 3ms, too fast for a network timeout, so
almost certainly local connection-pool exhaustion rather than anything the
targets did.

**4 of the 5 units recovered on retry. 1 did not**:
`ctx.delivery_town.d15` exhausted all four attempts, and that trial is lost. It
appears as `context UNSCORABLE=1` for `gpt-4o-mini` rather than silently
vanishing, and `gpt-4o-mini`'s context denominator is 35, not 36.

The run cannot say what those failures *were*. `error_excerpt` stores the head
of a failed response body, and a transport failure has no body, so the
exception was classified as `retryable` and its message discarded. That is
fixed for future runs; it could not be fixed retroactively for this one.

## What this run does not show

- **precision@k, recall@k, MRR and nDCG.** 144 SKIPPED rows, every run, with
  the reason attached. They need relevance labels over the target's own
  corpus; a black-box client cannot supply the corpus, enumerate it, or know
  what should have been retrieved. This is a permanent statement, not a
  "not yet".

- **Degradation's context-limit branch.** No probe here approaches any of
  these models' windows. It remains exercised only against the mock.

- **Induced tool failures**, degradation's third dimension: they require
  `tool_calling`, and the one model that would otherwise be interesting here
  rejects tools.

- **Whether `gpt-5-search-api` truly lacks tool calling.** The detector sends
  a request carrying a `tools` array, receives `404`, and reports
  `INCONCLUSIVE` with low confidence — correct from a single probe, since a
  `404` can mean the route is wrong rather than the feature absent. The run
  holds the control that would settle it: every other request to the same
  endpoint and model succeeded. The detector does not yet use it, so the
  published reason is weaker than the evidence supports.

- **Whether any cited page supports its claim.** Nothing fetches a cited
  source. Resolving a citation means issuing requests to third parties on the
  user's behalf, from a tool that promises it makes no network calls except to
  the target you name. "This URL exists" and "this page supports the claim"
  are both out of scope, and a contract test parses the retrieval module's
  imports to keep it that way.

## Reproducing this

```bash
sweepeval sweep --config sweepeval.openai-families.yaml --key $KEY \
  --i-am-authorized
```

Run id `20260919T075816521-99b26e`. `corpus_hash`
`62cbb800ecf96c889f113b8d1f866a0775d0eb97252f7376e26131f1fc974a80`.

Every figure on this page is checked against the run's own artifacts by
`scripts/verify_families_scorecard.py`, which re-derives them from
`observations.jsonl` and `aggregates.json` and fails if the document disagrees.
Prose figures in this project have gone wrong twice by hand and both times a
script caught what reading did not.
