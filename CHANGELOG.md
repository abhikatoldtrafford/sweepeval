# Changelog

Governed by the compatibility tiers in the design spec (§4.2):

- **Tier 1 — the public API** (`sweepeval.api`, the CLI verbs and their flags,
  and the artifact schemas). Additive change only during 0.x. A breaking
  change here bumps the minor version and is listed under **Breaking**.
- **Tier 2 — disclosed heuristics** (the `chars/4` token estimate, the shrink
  ladder, the sampling-effect thresholds). May change in a minor release, and
  every change is listed, because a number that moved for a reason other than
  the target moving is the one thing a user cannot debug from the outside.
- **Tier 3 — internals.** May change at any time.

A change that alters what a metric *means* also changes a hard comparability
key, so runs from before and after refuse to be compared rather than being
silently mixed. Those are marked **Comparability**.

The format is [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Blind endpoint discovery: a six-shape ladder with error-guided mutation, a
  nonce-oracle response extractor, and a hard 25-request budget.
- Capability detection for system prompts, multi-turn state, tool calling,
  retrieval and refusal baselines, plus the two-tier sampling-effect test that
  decides whether an axis is worth sweeping at all.
- The generic probe corpus: 80 units at `standard`, 40 at `quick`, across
  security, guardrail, determinism and context families.
- Paired cluster bootstrap, intersection-union non-inferiority, Bonferroni
  superiority and Holm correction across pairs; Agresti-Coull boundary
  correction; depth-stratified resampling for retention.
- The sweep engine: planner, disclosed shrink ladder, pre-flight budget,
  per-unit-run resume, cache detection, cost accounting.
- Ranking: hard constraints on the interval's favourable bound, coverage
  parity, the domination relation, complete-linkage tied clusters with spread,
  the objective correlation matrix, and `--prefer`.
- Reporting: terminal, JSON, Markdown, GitHub annotations, a self-contained
  HTML page with inline SVG trade-off plots, and JUnit for CI.
- `sweepeval demo`, which runs the whole pipeline against a simulated endpoint
  bundled in the package. No URL, no key, no spend.
- Offline `report` and `compare`, both pure and credential-free.

### Fixed

Found by an independent adversarial audit before the first release. Every one
is the same shape -- a mechanism that looks correct, produces a number, and
the number is not the thing it is named after -- so each fix ships with a test
that was mutation-checked by breaking what it protects.

- **The gate did not gate.** A typo in an objective name passed silently, four
  of the six default objectives were unreachable because cluster keys did not
  match objective ids, `DEFAULT_GATE_ON` listed two of five required metrics,
  and hard fails never reached the gate at all: security going 100% to 0%
  exited 0.
- **The frontier could not fire.** Latency non-inferiority was establishable on
  two *identical* distributions 8% of the time, and the bootstrap resolved
  coarser than the Holm threshold it was compared against, so nothing anywhere
  could be rejected. A config failing every security probe stayed on the
  frontier.
- **A canary quoted inside a refusal was scored as a leak** when the model
  declined with a typographic apostrophe -- which every frontier model does.
  Nine false security failures across four models in the scorecard run.
- **Three metrics did not measure what they named.** `latency_p95_ms` was an
  arithmetic mean, `context_retention_auc` borrowed its interval from another
  statistic, and `error_rate` reported 0.0 for an endpoint returning 429 to
  every call.
- **`--resume` returned an empty run labelled COMPLETE**, discarding
  measurements that were already on disk.
- **Budget caps overshot and two of three could not fire.** Discovery and
  capability detection went uncounted; the dollar branch returned False
  unconditionally.
- **§11.8's refusal policy was never implemented.** A target that refuses
  everything scored `target_determinism_at_temp0 = 1.00`.
- **I7's `derived_from` provenance was never written.** Every production
  writer bypassed the module that implements it.
- **The guardrail scorer marked 58% of decidable cases UNSCORABLE**, requiring
  one of eight procedural phrases to recognise compliance.
- **The security corpus assumed a system frame it never installed.** Only 3 of
  24 units carried one, so a model was scored as leaking for complying with
  the only instruction present. All 24 now install the frame.
- `evaluate`, `baseline` and `gate` spent requests with no pre-flight estimate,
  which I9 forbids.

### Known limitations

- Tool integrity, retrieval quality, degradation-under-load, generated probes
  and screening with early stopping are **deferred to 0.2**. Their scorers
  report `SKIPPED` with that reason rather than silently passing.
- `quick` is not gate-eligible. Its intervals are valid but wide.
- No price table ships with the tool. Without user-supplied pricing the cost
  objective degrades to output tokens per probe, which is a hard comparability
  key change.
