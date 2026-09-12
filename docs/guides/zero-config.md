# Zero-config walkthrough

What actually happens when you type this:

```bash
sweepeval run https://your-endpoint --key $KEY
```

## 1. Pre-flight, before anything is sent

You are shown an estimate covering **every** phase — discovery, capability
detection, scoring, the hard-fail confirmation allowance — and asked to
confirm. Nothing has been sent yet. Without a TTY and without `--yes`, the run
declines.

## 2. Discovery

A free metadata sniff first: `/.well-known`, an OpenAPI document, a model
list. Nothing billable.

Then the shape ladder. Six built-in shapes in prior-likelihood order —
OpenAI chat completions, Anthropic messages, Gemini generateContent, a bare
`{"prompt": ...}`, a bare `{"input": ...}`, raw text — each tried with the
single inert prompt `Reply with the single word OK.` Nothing in discovery asks
the target to act.

Every unmutated shape is tried before any mutated one. That ordering matters:
an OpenAI body with `max_tokens` added is a valid Anthropic request, so
without it a mutated near-miss can beat the exact match.

If a shape returns a 4xx with a usable error message, up to six **error-guided
mutations** are tried — rename a field, add a required one, wrap or unwrap the
body — at most two deep.

Auth is resolved once, not per shape. Four rungs across six shapes with
mutations exhausts a 25-request budget before reaching the third shape.

## 3. Extraction

Which field of the response holds the answer? The primary method is a **nonce
oracle**: send `Reply with exactly the following and nothing else: <nonce>`
and find the JSON path containing the nonce.

This exists because heuristics lose to echoed prompts. An endpoint that mirrors
your request text back in some field beats the real answer on every
length-and-position heuristic there is. The oracle does not care.

If the target will not comply, a blind walk falls back to scoring candidate
paths, penalising anything that looks like a near-duplicate of the request.

Discovery prints its confidence and writes an editable config. Correct it and
re-run if it guessed wrong.

## 4. Capability detection

Does a system role actually change behaviour? Not "is it accepted" —
*honoured*. A target that silently drops the role would otherwise get a
four-variant sweep axis producing four identical configurations.

Does multi-turn history work? Are tools available? Does the target have a
recognisable refusal fingerprint — needed so a refusal can be told from an
error?

Then the **sampling-effect test**: two open-ended prompts, three runs each at
a low and a high setting, per parameter. Tier 1 counts distinct outputs
against a fixed decision table; if that cannot separate, tier 2 runs a
dispersion test with a TOST equivalence check. A parameter proved inert is not
swept. An inconclusive one *is* — including an inert axis costs money,
excluding an effective one silently truncates the experiment, and the
asymmetry is deliberate.

## 5. Planning

Axes come only from what discovery proved is variable: usable model ids, the
system-prompt variants if the role is honoured, the sampling parameters the
test found effective. Each rejected axis is printed with its reason.

The cross product is shrunk to the profile's cap by the disclosed ladder.

If nothing survives, that is not an error: the run is a
single-configuration evaluation with a banner listing every candidate axis and
why it was rejected. For a custom agent system, that is the modal outcome.

## 6. Execution

Each configuration runs the frozen probe set N times. Turns are **scripted** —
a turn whose content depended on the previous answer would unfreeze the probe
set and break the pairing.

Multi-turn conversations are replayed statelessly, and a mid-conversation
failure restarts from turn 1 rather than resuming, because resuming would
diverge state on a server session.

Checkpoints are per unit-run, written only after the rows are durably
appended, so a crash leaves orphan rows that aggregation skips rather than a
checkpoint claiming work that was never stored.

## 7. Aggregation and ranking

Cluster tables per metric, intervals by paired cluster bootstrap, then
constraints, coverage parity, domination, Holm, tied clusters, correlation.

## 8. What you are left with

```
.sweepeval/runs/<run-id>/
  manifest.json      comparability keys, versions, hashes, seeds, capabilities
  plan.json          configs, axes, shrink steps, serialised units, canaries
  calls.jsonl        one row per HTTP call
  observations.jsonl one row per (config, unit, run, scorer, metric)
  aggregates.json    intervals and cluster tables -- enough to re-rank offline
  frontier.json      clusters, dominators, violators, every comparison
  blobs/<sha256>     content-addressed extracted text
  state/<config>.json completed unit-runs and budget counters
```

All of it re-reportable offline:

```bash
sweepeval report .sweepeval/runs/<run-id> --format html --prefer security,cost
```
