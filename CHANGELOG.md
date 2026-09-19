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

- **The `retrieval` family is built (§11, family 6) — the last deferred one —
  and its detector was wrong in the same way the tool one was.** The citation
  probe asked "What sources support your answer? Cite them." with no prior
  answer to support, then searched the response for
  `documents`/`sources`/`citations`/`retrieved`/`chunks`. Measured against
  `gpt-5-search-api`, a live search-backed model returning real citations: not
  one of those keys appears anywhere in its response. Its sources arrive as
  `message.annotations[].url_citation`, so a working retrieval endpoint was
  reported UNSUPPORTED, indistinguishable from a plain model. The probe now
  asks something weights alone cannot answer, and five channels are read.

  **precision@k, recall@k, MRR and nDCG are not computed, and never will be
  from here.** They need relevance labels over the target's own corpus;
  sweepeval cannot supply the corpus, enumerate it, or know what should have
  been retrieved. They are reported SKIPPED with that reason on every run --
  a different statement from "not built yet", and one the tests pin.

  **Nothing fetches a cited source.** Resolving a citation would mean issuing
  requests to third parties on a user's behalf, from a tool that promises it
  makes no network calls except to the target you name. So "this URL exists"
  and "this page supports the claim" are out of scope and the report does not
  imply otherwise. A contract test parses the module's imports to keep it that
  way.

  What is measurable: `citation_integrity` (sources surfaced when the question
  needs them, identifying something, with spans that land inside the answer --
  and nothing cited for a question nothing could source) and
  `citation_stability` (the same question surfacing the same sources twice,
  compared as a set, because ordering is ranking and ranking needs labels).

  Four of the twelve probes ask about a company that does not exist, a
  standard never published, an event that has not happened and a fact nobody
  could know. Live against `gpt-5-search-api`: every grounded probe passed
  with well-formed citations, it declined the invented company and the private
  fact -- and it cited real ISO catalogue pages for **ISO 99145-7, which does
  not exist**, on both runs. Citation stability failed on every grounded
  probe: the same question surfaces different sources run to run.

  Neither metric is a default objective, for the same reason as the other two
  late families.

- **The `tool_integrity` family is built (§11, family 4), and the capability
  detector that deferred it was wrong.** The detector sent a bare prompt --
  "if you have a tool available, call it" -- and looked for `tool_calls` in the
  reply, without ever offering a tool. A chat API only emits that field when
  the request carries a `tools` array, so `UNSUPPORTED` was the only answer it
  could give. Verified against api.openai.com: identical prompt, no tools
  offered -> no `tool_calls`; one tool offered -> `tool_calls` naming the right
  function. Every run this tool has done reported `tool_calling=UNSUPPORTED`
  against endpoints that support it, and this family was deferred partly on
  that reading.

  sweepeval now declares the toolkit itself and offers it -- it cannot know a
  target's own tool schema, and knowing the schema is exactly what makes "did
  this call conform" answerable. **Nothing offered is ever executed**; a tool
  call is a request to run something, and it is the request that is scored.

  Two metrics. `tool_call_validity` asks whether the right tool was called with
  arguments the schema will accept: an unknown tool name is a hallucination, a
  missing required argument is an incomplete call, a wrong JSON type is drift.
  Three of the twelve probes expect *no* call, because a target that reaches
  for a tool on every turn is as broken as one that never reaches for the right
  one, and it bills for the privilege. `tool_selection_stability` asks whether
  the same prompt picks the same tool across runs -- kept apart because they
  fail independently.

  Four reply encodings are read, because "supports tool calling" is not one
  wire format: the current `tool_calls` array, the deprecated `function_call`
  object, an Anthropic `tool_use` block, and a call written into the message
  text. The fourth is UNSCORABLE with the encoding named -- crediting prose as
  a tool call reports a capability the target does not have, and scoring it as
  silence loses the finding. A shape that cannot carry a tool declaration at
  all is NOT_PROBED rather than UNSUPPORTED.

  Live against gpt-4.1-nano: 22 of 24 valid, selection stable on all twelve
  probes -- and a pure arithmetic question ("48 cartons, 6 pallets") makes it
  call a tool, a different one on different runs.

  Neither metric is a default objective, for the same reason as degradation's.

- **The `degradation` family is built (§11, family 8).** It was one of three
  registering `SKIPPED: not_implemented`, and it went first of the three
  deliberately: `tool_integrity` and `retrieval` need capabilities every
  endpoint measured so far reports UNSUPPORTED, so their code would ship long
  before any real evidence about them could. This one needs none.

  Two dimensions, two metrics, no blending:

  `degradation_resilience` plants a fact, buries it under 1k-24k characters of
  generated filler and asks for it back. A body the target refuses for
  exceeding its context window is UNSCORABLE with that reason, not FAIL --
  "your window is smaller than this probe" and "you read it and forgot" are
  different findings and only one is about resilience.

  `load_resilience` asks the same question while eight requests are in flight,
  and scores it **against an identical probe answered alone**. The pairing is
  the whole metric: scored bare, the first live run against gpt-4.1-nano
  reported two load failures, and a strictly serial control reproduced both
  exactly -- that model answers "LINNET" for "LINNET-7704" whether or not
  anything else is in flight. A number in a family called `degradation` must
  not report a model's baseline mistake as damage done by load. Both sides
  failing is now UNSCORABLE, and says so.

  Induced tool failures, the third dimension, report SKIPPED: they need
  `tool_calling`.

  Neither metric joins the default frontier. §14.1's six dimensions are a
  decision, and widening them silently would make almost every config
  non-dominated for everyone. Opt in with `--objectives`.

  The ramp is the only place the tool contends with itself. It is bounded by
  `Governor.MAX_BURST_CONCURRENCY` (8) rather than by anything a corpus file
  can name, it is scoped to a block that restores the limit even on an
  exception, and its calls are excluded from the latency population -- the
  operational scorer scores every unit's calls, so without that the ramp would
  have quietly moved `latency_p95_ms` on every run that included this family.

  `standard` and `deep` only, and thirty probes rather than seven: each metric
  resamples over its own probes, so the bootstrap floor binds on each
  separately. The cluster-floor contract test caught the first draft.

- **`sweepeval rejudge` — the judge, on a run you already paid for (§11.9,
  §5.1).** `rejudge <run> --judge MODEL --judge-url URL` re-scores a stored
  run, escalates every observation a scoring contract marked ambiguous, and
  writes a new run. The judge decides from the response text and the response
  text is in the store, so re-running the whole sweep with `--judge` was the
  wrong price: on the published 14-model run this is 421 judge calls rather
  than 7,917 target requests. The count is exact rather than estimated — the
  ambiguities are on disk — and the pre-flight shows it before spending.

  A **new** run, not an edit. `observations.jsonl` is append-only and is what
  was paid for (I7); `judge` is a hard comparability key, so a judged result
  and an unjudged one are different measurements and `compare` must refuse to
  put them side by side. The derived run carries the source's blobs and plan,
  records `derived_from`, and its `calls.jsonl` holds the judge's calls and
  nothing else — zero target calls, which is the point of the price.

  What it showed, on the scorecard run: coverage from 25–38 of 60 to **60 of
  60** on every model, all fourteen `LOW_COVERAGE` flags cleared, 225 PASS and
  196 FAIL. The unjudged rates turn out to have been computed on a biased
  subset — the responses no lexical rule settles are exactly the hedged ones —
  and the ordering barely survives judging, Spearman ρ 0.35, with `gpt-4.1`
  falling 0.727 → 0.417. It does **not** resolve the ordering: no model pair
  separates on non-overlapping intervals either way.

  And it is **not reproducible**. Two judged passes over byte-identical stored
  text, same model, same prompt version, `temperature: 0`, disagreed on 21 of
  420 verdicts — 5.0%. §11.9 asks for a pinned model, zero temperature and a
  fixed prompt and gets all three; determinism is not what they deliver. The
  published intervals are over probe clusters and do not include that
  variance. `scripts/judge_stability.py` measures it;
  `scripts/verify_judged_scorecard.py` checks all 94 published figures against
  the runs.

- **The model is targetable and recorded (§16).** `evaluate`, `baseline` and
  `gate` take `--model`; `baseline.json` gains a `model` field; the gate
  refuses a run whose model is not the baseline's, exit `2`, naming both.
  `--allow-model-change` compares them anyway and annotates the verdict.

  `baseline` and `gate` could not target a model and did not record one. The
  model came from discovery's preference heuristic -- first id on `/v1/models`
  containing `mini`, `flash`, `haiku`, `small`, `lite` or `turbo` -- and
  nothing wrote it down: the manifest's target block names the url, shape,
  auth and extraction path; `calls.jsonl` carries a `params_hash` and no model
  string; the baseline carried `config_id: "default"`. So a committed baseline
  did not say what produced it, and a change to a provider's model listing
  could move a CI gate onto a different model between the baseline run and the
  gate run with no signal anywhere. Measured against `api.openai.com`: a
  gpt-5-mini baseline gated against gpt-5-nano exited 1 on `cost_per_probe`
  (1881 -> 3074, p=0.0005) and never mentioned the model.

  The model is **not** a comparability key and cannot become one -- a sweep
  varies it across configs inside one run, and a run carries one comparability
  block, so a per-run key would have to lie for every sweep. It is checked
  where the pairing actually lives, between a baseline and its gate.

  `gate` defaults to pinning the baseline's model, so it re-measures what the
  baseline measured rather than spending a full run -- 120 requests at
  `standard` -- on a comparison it can only refuse. A pin that the shape
  cannot carry is refused outright (exit `3`) rather than accepted and
  ignored, which would put a model id in a committed baseline that no request
  ever named. Baselines written before this release still load; the gate says
  it could not check.

- `gate.json` carries `model: {baseline, current}`, and the single-config
  `report.json` carries `model`. A gate log read three weeks later could not
  say which models the verdict was about.

- **`max_configs` in `sweepeval.yaml`.** `profile` and `runs` were
  file-settable and the config cap was not, though it moves the bill further
  than either. A declared axis does not displace the discovered ones -- the
  planner crosses them -- so a file declaring four models still planned twelve
  configurations, and a comment reading "model is the only axis" did not make
  it one. An explicit `--max-configs` still wins over the file.

### Fixed

- **Capabilities were detected once per sweep and applied to every config.**
  Capabilities are a property of the model, and a sweep's whole point is
  varying the model, so the report was built against whichever model discovery
  happened to pick and then asserted of all the others. In a four-model run
  that cost both directions at once: `gpt-5-search-api`, the one model present
  with retrieval, was told it had none and the family was skipped for every
  config -- zero retrieval rows from a run whose purpose was retrieval
  evidence -- while it was simultaneously told it had tool calling, which it
  rejects, so 36 probes went out and came back 404 and its 48 tool rows read
  UNSCORABLE rather than honestly skipped. Each config now probes its own
  model and gates its own unit set; a config pinning no model reuses the
  run-level report rather than paying twice. The extra phase is a separate
  line in the pre-flight, outside the unavoidable floor -- folded into it, a
  cap that used to buy a partial sweep began declining the run outright.

- **The artifact still recorded one capability report for the whole run**,
  one commit after the runtime learned to detect them per model. No capability
  block and no skip list was serialized per config, and run-level `skipped`
  was derived from the run-level report -- so a sweep was on course to write
  `retrieval SKIPPED -- UNSUPPORTED` beside observations containing retrieval
  rows. An artifact that contradicts its own observations is worse than one
  that omits the claim, and which family was gated off for which model could
  not be read back at all. Each config payload now carries its own verdicts
  and its own skip list, and run-level `skipped` is their intersection: a
  family appears there only when no config scored it. I5 is not weakened --
  a family no config could score is still reported, with the reason.

- Cross-run scoring excluded any run with no extracted text, which is every
  successful run of a tool-calling probe: a reply carrying only tool calls has
  `content: null`. `tool_selection_stability` came back "fewer than two
  scorable runs" on every probe that had worked. A family can now opt into
  comparing runs that produced no text; the default is unchanged, because for
  determinism and context an empty response really is nothing to compare.

- A failed conversation was recorded under `f"{family}_pass_rate"`: a real
  metric for security and guardrail, and a name nothing declares for the rest.
  A failed `context` conversation went to `context_pass_rate`, which no scorer
  emits, no aggregator reads and no coverage counter counts -- so the row
  vanished, which is the exact outcome the surrounding code exists to prevent.
  The metric is now asked of the family's scorer.

- A re-scored observation lost its `blob_ids`. A scorer is handed text rather
  than blob addresses, so the row it returns names none, and nothing noticed
  while a re-score was only ever a report. Persisting those rows — which
  `rejudge` does — produced a run whose every verdict had no evidence behind
  it: 1,422 unjoinable rows and a run that could not be re-scored again.

- `rescore` re-scored the judge's own rows with the deterministic scorer,
  which is precisely the scorer that could not settle them. On a judged run
  that reported 420 spurious changed verdicts and marked every config as
  touched, leaving the reproduction self-check with nothing to check.

- A swept or pinned `model` against a Gemini target went into
  `generationConfig.model` -- a field the API does not read, in a body it
  rejects -- because `model` is a sampling key for the shapes that carry it in
  the body and Gemini names it in the URL. Every request of a Gemini sweep
  with a model axis was affected.

### Changed

- The pinned params live on `LadderResult`, so every body built from a ladder
  inherits them. Pinning on the probe plan alone left capability detection --
  five detectors and the sampling probe, all building bodies straight off the
  ladder -- probing a different model from the one the metrics came from, which
  matters precisely where it is worst: a reasoning model rejects the sampling
  parameters a chat model accepts.

### Comparability

- The corpus gained the degradation, tool-integrity and retrieval families, so
  `corpus_hash` changes. Runs from before and after refuse to be compared rather than being silently
  mixed, and a committed `baseline.json` predating this release will need
  re-taking. `quick` is unaffected in content but shares the hash.

## [0.2.0] - 2026-09-14

### Added

- **LLM-judge escalation (§11.9).** Off by default; `--judge MODEL
  --judge-url URL`. It resolves the ambiguities a deterministic contract
  declared it could not settle, and nothing else -- the trigger is a scoring
  contract's own `ambiguous_when`, so the judge never sees a case a regex got
  right.

  It exists because retuning the guardrail scorer against 60 live responses
  left 45% that no lexical rule can classify. Not edge cases: hedged, general,
  helpful answers, which is most of what a good model says when asked for
  something it should withhold. Measured against a live gpt-4o-mini judge, the
  guardrail metric goes from `NO_VALID_INTERVAL` at 0/20 coverage to
  `1.000 [0.596, 1.000]` at 14/20.

  What it refuses to do is most of the design. It will not score its own
  output: a judge model that is one of the models under test is refused at any
  endpoint, and a judge sharing an endpoint with a target whose model is
  unknown is refused too. It will not guess: anything that is not strict JSON
  with one of three verdicts is a recorded failure, and the deterministic
  UNSCORABLE stands. It will not vary: temperature 0, pinned model, versioned
  rubric, all three in the manifest.

  The probe and response are fenced and labelled as data in the prompt,
  because a probe in this corpus is an adversarial injection string by design
  and the judge is an LLM -- scoring a successful injection must not mean
  running it.

  Verdicts are appended as a second observation rather than replacing the
  first, so the log keeps both and any model-decided number can be dropped by
  filtering on `scorer == "judge"`.

- `--max-tokens` and `--max-dollars` on `sweep`. All three cap units bound in
  the engine, and only requests was reachable from the CLI -- which is why the
  bugs in the other two survived so long.
- `latency_p90_ms`, registered and promotable: §13.3's documented fallback for
  a p95 that needs 72 probes before a distribution-free bound exists.
- Third-party scorers and objectives actually load from their entry points.

### Fixed

- **A re-scored trial is one trial.** The judge appends rather than
  overwrites, and both coverage counters counted rows -- so a judged family
  with 20 trials reported 14 scored of 34, understating coverage exactly where
  the judge had improved it.
- The judge independence check refuses self-judging rather than shared
  hosting. §11.9's literal endpoint rule made the judge unusable for sweeping
  models on one provider, which is the commonest configuration there is.
- The guardrail scorer no longer calls correct behaviour a policy breach.
  Retuned on live responses; see below.

### Changed

- The deferred families no longer name a release. Their scorers said
  `not_implemented_in_v0.1` and their briefs said the work would "land in
  v0.2" -- and this *is* 0.2, shipped without them, so the tool announced a
  broken promise on every run in a machine-readable string a user could have
  planned around. The reason is now `not_implemented` and the briefs say
  "specified but not built". Stored observations keep the string they were
  written with, so an offline re-report of an older run is unchanged.

### Comparability

- `security` and `guardrail` scorers are at **v2**. Both changed what their
  number means, so `scorer_versions` refuses to compare a v1 baseline rather
  than mixing definitions.
- `judge{present, model, prompt_version}` is populated. A judged run refuses
  to compare against an unjudged one, and editing a rubric refuses too.

## [0.1.0] - never released

Everything below was built and merged under this version, and the two
independent audits that followed found enough wrong with it that no release
was cut. It is kept as its own section because the **Fixed** list under it is
the record of what those audits found, and because the version numbers in
`scorer_versions` refer to it.

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

A second independent audit, after those fixes, found seven more. All are
fixed; each is listed because the tool's whole claim is that it says what it
did not measure.

- **`security_pass_rate` scored an unreadable target 1.00.** It was the only
  family with no UNSCORABLE branch, so "no text extracted" read as "no leak
  found" — with a real interval and a coverage line reporting 20/20 scored.
- **A refusal cue could be any of several bare phrases matched as
  substrings**, so a canary emitted verbatim scored a pass if the model
  apologised within 120 characters of it, and a full PII leak passed if it
  opened with "I'm sorry to hear that". Cues are now first-person anchored
  patterns, and the exclusion window stops at a sentence boundary.
- **A partial `--resume` reported determinism 0.00 where an uninterrupted run
  reported 1.00**, propagated it to temperature siblings, wrote both answers
  to the append-only log, and called the run COMPLETE.
- **The CI gate compared the unweighted mean of every objective**, so a
  29-point `context_retention_auc` regression exited 0 and the two numbers
  printed beside the verdict were means under the AUC's name. `Baseline` now
  carries `strata`, and the gate and the frontier share one statistic dispatch.
- **A gate that could test one of five requested metrics exited 0 silently.**
  It now reports `not_gated` and `degraded` in the terminal, the JSON payload
  and GitHub annotations.
- **A hard fail required a leak on every run**, where D23 asks for a majority
  of three — so a target exfiltrating its system prompt on two attempts in
  three was neither disqualified nor reported anywhere. Suspected leaks are
  now printed too.
- **The cluster floor bound the interval but not the p-value.** Below eight
  clusters `paired.py` bootstrapped anyway; measured family-wise false
  domination was 8.5% against a stated 5%. It is now an exact sign-flip
  permutation test, and the measured rate is 0%.

### Withdrawn

- **The OpenAI scorecard's security figures and frontier.** Produced by the
  canary matcher above, wrong in the unsafe direction, and not re-scorable
  from the stored run: `blob_ids` was empty on every observation, so no stored
  response could be joined back to its probe. Both fields are populated now,
  so a run recorded today re-scores for free. The operational table stands and
  now prints its intervals.

### Known limitations

- Tool integrity, retrieval quality, degradation-under-load, generated probes
  and screening with early stopping are **specified but not built**. Their
  scorers report `SKIPPED` with that reason rather than silently passing.
- `quick` is not gate-eligible. Its intervals are valid but wide.
- No price table ships with the tool. Without user-supplied pricing the cost
  objective degrades to output tokens per probe, which is a hard comparability
  key change.
