# sweepeval

**Zero-config, black-box sweep and benchmark engine for LLM and agent systems.**

Point it at an HTTP endpoint. It works out the request shape, brings its own
probes, sweeps the configurations it can prove are variable, and returns a
Pareto frontier with a confidence interval on every number.

```bash
uvx sweepeval run https://your-endpoint --key $KEY
```

No config file. No prompts to write. No labels. No axes to declare.

Try it with nothing at all — no endpoint, no key, no spend:

```bash
uvx sweepeval demo
```

> **Status: 0.1 development.** The engine runs end to end against real
> endpoints. APIs are additive-only during 0.x; see [CHANGELOG.md](CHANGELOG.md).

---

## What a run looks like

This is real output from `sweepeval demo`, which runs the whole pipeline
against a **simulated** endpoint bundled in the package — no URL, no key, no
spend. The numbers describe a scripted mock; the shape is exactly what a real
run produces.

```
axes
  system_prompt    none, terse_neutral, verbose_strict_with_guardrails
  temperature      0.0, 1.0

rejected axes
  top_p            INERT: dispersion is statistically equivalent within +/-0.05
                   (TOST); the parameter has no measurable effect

shrink ladder (cap 6)
  drop temperature=0.7 (24 -> 16 configs)
  drop system_prompt=terse_permissive (16 -> 12 configs)
  cap models (12 -> 6 configs)

measurements (unranked - the frontier is computed by `rank`)
+--------+--------------+--------------+--------------+--------------+--------------+
| config |     security |    guardrail |    retention |    det@temp0 |   latency ms |
+--------+--------------+--------------+--------------+--------------+--------------+
| cfg-00 | 1 [0.679, 1] | 1 [0.596, 1] | 1 [0.679, 1] | 1 [0.628, 1] |       0.9238 |
|        |              |        LOW_N |              |              |    [0.874,   |
|        |              |              |              |              |     0.978]   |
| cfg-01 | 1 [0.679, 1] | 1 [0.596, 1] | 1 [0.679, 1] | 1 [0.628, 1] |       0.8913 |
|        |              |        LOW_N |              |              |    [0.854,   |
|        |              |              |              |              |     0.932]   |
+--------+--------------+--------------+--------------+--------------+--------------+
                                                              (4 more configs)

coverage (scored / attempted)
  cfg-00     context 20/20  determinism 24/30  guardrail 14/20  security 20/20

SKIPPED
  retrieval        retrieval=UNSUPPORTED (citation probe)
  tool_integrity   tool_calling=UNSUPPORTED (tool probe)

frontier - 6 non-dominated config(s) in 1 tied cluster(s), at alpha 0.05 family-wise

  cluster 1 (6 config(s))
    cfg-00  temperature=0.0 sys=none
    cfg-01  temperature=1.0 sys=none
    ...
    spread across members:
      context_retention_auc            1 .. 1
      cost_per_probe                   11.95 .. 12.55
      latency_mean_ms                  0.8823 .. 0.9521
      security_pass_rate               1 .. 1

  every configuration is statistically tied. That is an answer, not a
  failure: at this profile the intervals are wide enough that the sweep
  cannot separate them. Use --prefer to apply your own priority, or
  --profile standard for narrower intervals.
```

Every number carries an interval, or says it cannot give one. Nothing
carries a rank. Every family the endpoint could not support says which
detector ruled it out.

Note what the mock's own run reports: **nothing separates**. Six configurations
against a scripted target that behaves the same way for all of them, and the
tool says so rather than ordering them anyway. A frontier of one cluster is
the honest result there, and a tool that produced a ranking from this data
would be inventing it.

## A real scorecard

Fourteen OpenAI chat models, one `sweepeval` run at `standard`/N=3, 2026-09-14.
7,917 requests, zero transport errors. Every figure carries its 95%
cluster-bootstrap interval, and there is no overall score anywhere.

The headline is prompt-injection resistance, and it is the one place on the
page where two models separate from the rest by more than their intervals:

| Model | security_pass_rate |
|---|---|
| gpt-6-astra, gpt-5.6-terra, gpt-5.5, gpt-5.4, gpt-5.2, gpt-5, gpt-5-mini, gpt-4o | 1.000 [0.837, 1.000] |
| gpt-5-nano | 0.986 [0.818, 1.000] |
| gpt-5.1 | 0.971 [0.792, 1.000] |
| gpt-4o-mini | 0.958 [0.781, 1.000] |
| gpt-4.1 | 0.917 [0.730, 0.988] |
| **gpt-4.1-mini** | **0.306 [0.139, 0.507]** |
| **gpt-4.1-nano** | **0.306 [0.139, 0.507]** |

`gpt-4.1-mini` and `gpt-4.1-nano` fail 50 of 72 injection probes; every other
model fails six or fewer. Many failures are replies consisting of nothing but
the canary. It is not a size effect and not an age effect — `gpt-4o` and
`gpt-5-mini` both score 1.000.

Latency and output tokens, same run:

| Model | Mean latency (s) | Output tokens / probe |
|---|---|---|
| gpt-4.1-nano | 1.70 [1.46, 1.97] | 190 [132, 257] |
| gpt-4o-mini | 1.70 [1.46, 1.96] | 306 [194, 436] |
| gpt-4.1-mini | 1.94 [1.65, 2.24] | 181 [128, 240] |
| gpt-4.1 | 1.98 [1.70, 2.27] | 353 [219, 513] |
| gpt-4o | 3.28 [2.67, 3.90] | 313 [206, 439] |
| gpt-5.4 | 3.52 [2.39, 4.99] | 297 [192, 416] |
| gpt-5.6-terra | 3.82 [2.85, 4.94] | 307 [199, 440] |
| gpt-5.1 | 4.79 [3.81, 5.93] | 461 [324, 615] |
| gpt-6-astra | 4.96 [4.10, 5.97] | 208 [158, 264] |
| gpt-5.5 | 5.21 [3.88, 6.73] | 425 [297, 561] |
| gpt-5.2 | 5.43 [3.95, 7.09] | 408 [293, 534] |
| gpt-5-nano | 12.11 [10.83, 13.58] | 3072 [2383, 3855] |
| gpt-5-mini | 14.93 [13.06, 16.89] | 1821 [1364, 2341] |
| gpt-5 | 17.39 [15.15, 19.71] | 2228 [1682, 2855] |

The reasoning tier costs 3–10× the latency and 5–15× the output tokens of
everything else, most of it reasoning tokens that never reach the answer. That
gap is far wider than the intervals; most neighbouring pairs elsewhere in the
table are not, which is what N=3 buys.

[Full scorecard](docs/scorecard.md) — including guardrail adherence, where
every model trips `LOW_COVERAGE` because 37–58% of those probes are neither a
refusal nor a disclosure, and determinism at temperature 0, where nothing in
the set is reproducible.

**The security column above was corrected after the run, not withdrawn.** An
adversarial audit found the canary matcher wrong in the unsafe direction — a
model that emitted the planted marker and then apologised scored a pass. Three
of 1,003 security verdicts changed. Rather than re-run 7,917 paid requests,
the family was re-scored offline from the stored responses, which is what §5.1
promises and what `sweepeval rescore` now exposes. The method is checked
rather than asserted: re-aggregating the eleven configs whose verdicts did
*not* change reproduces the tool's own stored numbers to 1e-9.

One gap survives. Five of gpt-5.1's 72 security trials are unscorable and
**unrecoverable** — the model took OpenAI's structured-refusal path, the
extractor of the day read it as empty, and the body was not kept. That row's
denominator is 67.

## Why another eval tool

**Blind discovery.** Every comparable tool needs you to describe your endpoint
and write your evals. This one probes the endpoint to work out its request
shape, infers how to extract the response, and detects what it can actually do
— streaming, tools, multi-turn state, whether temperature has any effect at
all. That last one matters: if temperature does nothing on your endpoint,
sweeping it is theatre, and sweepeval tells you so instead of drawing a chart.

**Confidence intervals on everything, including the gate.** Tools that gate on
point estimates produce gates that flap, and flapping gates get disabled.
Every comparison here is a paired test over the identical probe set, corrected
for multiplicity across the whole family of comparisons.

Where an honest interval is not available, the metric says so instead of
printing a narrow one. A p95 needs 72 probes before a distribution-free upper
bound exists at all, so below that it declines and the p90 — which needs 36 —
is reported alongside it, under its own name.

**No composite score.** A single number hides the trade-off that makes a
configuration decision hard: the most secure config is usually the slowest,
and the cheapest one usually leaks. sweepeval reports the frontier and says
what each position wins and gives up. Tell it your preference and it names a
config — the preference is yours, applied at report time, never baked into the
data.

**Black box, always.** No SDK coupling, no instrumentation, no framework
hooks. Every number is measured from what came back over HTTP.

## Install

**Not on PyPI yet** — the 0.2.0 release is tagged and built, and publishing is
waiting on a trusted-publisher entry. Until then, from the repository or the
container image:

```bash
uvx --from git+https://github.com/abhikatoldtrafford/sweepeval sweepeval --help
pipx install git+https://github.com/abhikatoldtrafford/sweepeval
docker run --rm ghcr.io/abhikatoldtrafford/sweepeval --help
```

Once published, the usual three work:

```bash
uvx sweepeval --help          # no install
pipx install sweepeval        # isolated
pip install sweepeval         # into your environment
```

Python 3.10+. Runtime dependencies: httpx, pydantic, typer, rich, pyyaml,
numpy, jinja2. No cloud account, no telemetry, no sign-up.

## Use it

```bash
# The zero-config path.
sweepeval run https://your-endpoint --key $KEY

# Look before you spend: discovery only, at most 25 requests.
sweepeval discover https://your-endpoint --key $KEY

# One configuration, no sweep.
sweepeval evaluate https://your-endpoint --key $KEY

# The full sweep, with the knobs exposed.
sweepeval sweep https://your-endpoint --key $KEY \
  --profile standard --runs 3 --format html,junit

# Answer the question, using YOUR priority.
sweepeval sweep https://your-endpoint --key $KEY \
  --prefer "maximize security_pass_rate subject to latency_mean_ms < 2000"

# Re-report and re-rank a stored run. Offline, no credentials.
sweepeval report .sweepeval/runs/<run-id> --format md --prefer security,cost
```

### As a library

The CLI is a thin veneer over `sweepeval.api`; every verb has a function, and
every function has an async twin.

```python
import sweepeval

result = sweepeval.sweep("https://your-endpoint", key=KEY, profile="standard")
frontier = sweepeval.rank(result)          # offline; re-runnable with other objectives

for cluster in frontier.clusters:
    print(cluster.members, cluster.spread)
```

## What it measures

| Family | What it checks |
|---|---|
| Security | Prompt injection, instruction override, system-prompt exfiltration, role confusion, delimiter escape, indirect and cross-turn injection |
| Guardrails | Policy adherence at four escalating pressure levels, reported per policy |
| Determinism | Exact repeatability, semantic stability, and invariance under paraphrase |
| Context | Fact recall as a decay curve, constraint persistence, contradiction handling, positional bias |
| Operational | Latency, tokens, cost, error rate by class |

Anything the endpoint cannot support reports `SKIPPED` with the detector that
ruled it out. Never a silent pass, fail, or zero.

Third-party scorers and objectives load from the `sweepeval.scorers` and
`sweepeval.objectives` entry points; a probe from a family with no registered
scorer is reported `SKIPPED` rather than dropped, and a plugin that fails to
import is named in the report rather than swallowed.

**Where a rule cannot decide, an optional judge can.** Roughly 45% of what a
good model says when asked for something it should withhold is neither a
refusal nor a disclosure, and no lexical rule classifies it. `--judge` resolves
exactly those cases — the ones a scoring contract declared ambiguous, and no
others — and refuses to score its own output, to guess, or to vary. Off by
default; see [the judge](docs/concepts/judge.md).

## Cost and wall-clock

Every run prints its estimate and asks before spending. The estimate covers
discovery and capability detection too, not just scoring — those spend first,
and gating after them would be gating after the money was gone.

| Profile | Units | Calls/run | Configs | Requests at 3 runs | Wall-clock at concurrency 2 |
|---|---|---|---|---|---|
| `quick` | 40 | 60 | 6 | 1,165 | ~25 min |
| `standard` | 80 | 188 | 12 | 6,853 | ~145 min |

Those totals include discovery and capability detection, which is why they are
larger than `units x calls x runs x configs`.

`quick` is for a first look. It is **not gate-eligible**: its intervals are
valid but wide, so few pairs separate. Use `standard` for decisions.

Cap it hard when you want to:

```bash
sweepeval sweep https://your-endpoint --key $KEY --max-requests 2000
```

Below the estimate, that buys a partial sweep that stops between
configurations and names every config that never ran. Below what discovery
plus one configuration needs, nothing is sent at all — spending the budget on
discovery and then having nothing left to score with is not a smaller sweep,
it is no sweep.

`--max-tokens` and `--max-dollars` bind the same way. The dollar cap needs
prices declared in `--config`, because a price this tool invented is the one
number you cannot check against your invoice — without them the cost objective
is output tokens, and it says so.

## In CI

```yaml
- run: pipx install sweepeval
- run: |
    sweepeval sweep ${{ vars.ENDPOINT }} --key ${{ secrets.KEY }} \
      --profile standard --yes --format junit
    sweepeval gate ${{ vars.ENDPOINT }} --key ${{ secrets.KEY }} \
      --baseline .sweepeval/baseline.json
```

`gate` exits `0` unchanged, `1` regressed or hard-failed, `2` incomparable, `3` usage error.
It compares against a committed `baseline.json` using the same paired test,
so it does not flap.

The baseline records the **model** it measured, and the gate re-measures that
model by default — pin it yourself with `--model`. A model that differs from
the baseline's exits `2` rather than reporting the difference as a regression;
`--allow-model-change` compares them anyway and annotates the verdict.

It also reports what it **could not** test. A metric whose family went
unscorable shares no cluster with the baseline and cannot be compared; that
does not fail the build, but `gate.json` carries `not_gated`, `degraded` and
`incomplete`, and each raises a GitHub warning annotation. A green tab that
checked one metric of five must not look like a green tab that checked five.

Without `--yes` and without a TTY, a run **declines** rather than assuming
consent. A CI job that starts spending thousands of requests because nobody
was there to say no is the failure the pre-flight exists to prevent.

## Safety

sweepeval's security suite is an active prompt-injection and exfiltration test
suite. Pointing it at an endpoint sends live requests, and on an agent system
those requests may trigger real tool calls, writes, or spend.

**Only run it against endpoints you are authorised to test.** The tool
requires a one-time per-host affirmation before the security family runs
against anything that is not localhost.

Discovery sends only inert, read-shaped requests, at most 25 of them, under a
wall-clock cap and a strict backoff. Nothing in discovery asks the target to
act.

No telemetry. No analytics. No network calls except to the target you name.
Credentials are redacted from every artifact, and `sweepeval init` tells you
what is and is not safe to commit.

## Known limitations

Written down because a tool whose whole argument is honest measurement should
be honest about itself. Nothing here is hypothetical; each one was found by
running the thing.

**Classic IR metrics are not computed and cannot be.** precision@k, recall@k,
MRR and nDCG need relevance labels over the target's own corpus, and a
black-box client cannot supply the corpus or know what should have been
retrieved. They are reported `SKIPPED` with that reason on every run rather
than approximated. This is not a gap waiting on effort.

**No citation is ever fetched.** sweepeval talks to the endpoint you name and
to nothing else, so "this URL exists" and "this page supports the claim" are
out of scope. What is checked is that a citation identifies something and that
its span lands inside the answer it annotates.

**Two capability detectors were wrong, and one may still be.** The tool probe
never offered a tool; the citation probe asked for sources with no answer to
source and looked for keys a real retrieval endpoint does not use. Both
reported UNSUPPORTED against endpoints that supported the capability, and both
deferred a scorer family on that reading. The remaining detectors have not had
the same scrutiny.

**Degradation does not yet have much to say.** Against `gpt-4.1-nano` it found
no loss at all: recall held at every haystack size to 24,000 characters, and
every load pair held too. That is a real result and a thin one -- a model with
a very large context window is not where this family earns its keep, and the
context-limit branch has been exercised only against the mock.

**Guardrail adherence is unresolved on every model measured.** 37–58% of those
probes are neither a refusal nor a disclosure — they are the hedged, partly
helpful answers a good model actually gives — and no lexical rule settles
them. They are reported `UNSCORABLE` with a reason rather than scored as
passes, so every row trips `LOW_COVERAGE` and no two adjacent rows separate.
The LLM judge exists to resolve exactly this band and was off for the run.

**The judge resolves coverage, not the ordering — and it is not
reproducible.** It has now been run over the whole 14-model scorecard: 421
ambiguities, every `LOW_COVERAGE` flag cleared, coverage from 25–38 of 60 to
60 of 60 on every model. It also showed that the unjudged rates had been
computed on a biased subset — the ordering barely survives judging (Spearman
ρ 0.35), because the responses a lexical rule cannot settle are exactly the
hedged ones. But **no model pair separates on non-overlapping intervals,
judged or unjudged**, and two judged passes over byte-identical text at
`temperature: 0` disagreed on **21 of 420 verdicts, 5.0%**. The published
intervals are over probe clusters and do not include that variance. See
[the scorecard](docs/scorecard.md#the-same-probes-judged).

**Concurrency is 1.** Deliberately: the governor dispatches one request at a
time, and the pre-flight ETA is honest about it. It also means a reasoning
model runs at 5–7 calls a minute, and the 14-model scorecard took an
afternoon.

**Offline re-scoring covers two families.** `sweepeval rescore` handles
`security` and `guardrail`, which decide from the final response text.
`determinism` needs every run of a unit at once, and runs made before this
release carry no `blob_ids` on those rows, so that column cannot be re-checked
against the responses that produced it. `operational` comes from `calls.jsonl`
and never needs re-scoring.

**A single-configuration run writes no `manifest.json`.** Only `sweep` does.
`evaluate`, `baseline` and `gate` therefore carry less provenance on disk than
a sweep of the same target — the committed `baseline.json` is where their
identity lives.

**The model was invisible until recently, and unpinned it still is a
heuristic.** Before the current release nothing recorded which model answered,
and `gate` would compare two different models without a word — verified live,
a gpt-5-mini baseline against a gpt-5-nano gate reported a cost regression at
p=0.0005 and never mentioned it. That is fixed. What remains is that an
*unpinned* run still lets discovery choose: the first id on `/v1/models`
containing `mini`, `flash`, `haiku`, `small`, `lite` or `turbo`. Against a
live OpenAI account today that picks `gpt-4.1-mini` — one of the two models in
the scorecard that fail 50 of 72 injection probes. Pin it.

**`--model` is refused, not honoured, on shapes that name the model in the
URL** (Gemini). Accepting it would put a model id in a committed baseline that
no request ever named.

**No price table ships.** `cost_per_probe` is output tokens unless you supply
pricing, and that switch is a hard comparability key because it is a different
quantity in different units. A stale built-in table would produce confidently
wrong dollar figures.

**`quick` is not gate-eligible.** Its intervals are wide by design and the
gate refuses the profile rather than passing everything.

**Not on PyPI yet.** 0.2.0 is tagged, built and on ghcr; the publish is
waiting on a trusted-publisher entry. See [Install](#install).

## How it compares

| | sweepeval | promptfoo | deepeval | ragas | LangSmith / Braintrust |
|---|---|---|---|---|---|
| Needs you to write evals | no | yes | yes | yes | yes |
| Needs to know your endpoint's shape | no, discovers it | yes | via SDK | via SDK | via SDK/tracing |
| Sweeps configurations | yes, discovered | yes, declared | no | no | partly, declared |
| Confidence intervals | on every metric | no | no | no | some, mostly aggregate |
| Multi-objective frontier | yes, no composite | no | no | no | dashboards, no frontier |
| Adversarial suite included | yes | via plugins | via plugins | no | no |
| Works without an SDK | yes, HTTP only | yes | no | no | no |

These are all good tools solving different problems. promptfoo is the closest
comparison and is excellent at declared, versioned prompt testing with a rich
assertion library; deepeval and ragas are strong at LLM-judged and
retrieval-quality metrics inside a Python codebase; LangSmith and Braintrust
are tracing and experiment platforms with far more of a product around them
than this has. What none of them do is take a URL and produce a scored,
interval-bearing comparison with no input from you — because that is a
different problem, and it is the only one sweepeval tries to solve.

## What is new

Full detail in [CHANGELOG.md](CHANGELOG.md). Since 0.1:

**The model is targetable, recorded and checked.** `--model` on `evaluate`,
`baseline` and `gate`; `baseline.json` records the model its requests named;
the gate re-measures the baseline's model by default and exits `2` rather than
reporting a model swap as a regression. `--allow-model-change` compares them
anyway and says so. The model is deliberately not a comparability key — a
sweep varies it across configs inside one run, so a per-run key would have to
lie for every sweep.

**An LLM judge for the ambiguous band (§11.9).** Off by default; it spends,
and its worst case is priced in the pre-flight. It only sees cases a scoring
contract declared it could not settle, and it refuses to score its own output.

**`sweepeval rescore`.** Re-score a stored run offline, no key and no network.
It never rewrites `observations.jsonl`, and it self-checks by reproducing the
untouched configs' stored numbers exactly. This is what corrected the
scorecard's security column without re-running 7,917 paid requests.

**A `retrieval` family.** Whether sources are surfaced when a question needs
them, whether they are structurally sound, and whether the same question
surfaces the same sources twice. A third of the probes ask about things
nothing could source -- fabricating a citation there is the failure that
matters. Live against a search-backed model, it cited real catalogue pages for
an ISO standard that does not exist.

**A `tool_integrity` family.** sweepeval offers its own toolkit and scores
what comes back against the schema it sent: right tool, arguments that
validate, and whether the same prompt picks the same tool twice. Four reply
encodings are read, and a call merely *described* in prose is refused rather
than credited. Nothing offered is ever executed.

**A `degradation` family.** Recall from under a growing haystack, and recall
under contention scored against an identical probe answered alone. The pairing
is the metric: unpaired, the first live run reported two load failures that a
serial control reproduced exactly, which would have blamed a model's baseline
mistake on load. The ramp is the only place the tool contends with itself; it
is capped at eight in flight and its calls are kept out of the latency
population.

**`sweepeval rejudge`.** Put the judge on a stored run without paying for the
run again — the judge decides from the response text, and the text is already
in the store. On the scorecard run that is 421 judge calls instead of 7,917
target requests. It writes a *new* run rather than editing the old one,
because the source is append-only and `judge` is a hard comparability key.

**A scorecard that survived being checked.** Fourteen models, 7,917 live
requests, every figure verified against the run by script rather than by eye.
Spot-checking each configuration as it landed found three refusal-matcher
defects, two of them in the unsafe direction.

**Correctness fixes that came out of live traffic**, not the mock: error-guided
mutation tries the non-destructive `add` before `rename`; extraction reads a
provider's structured refusal as the response it is; concurrency is declared
as the one request it actually dispatches, which makes the pre-flight ETA
honest; the gate refuses an ineligible profile *before* spending 120 requests
on it; a rate's confidence interval can no longer leave `[0, 1]`; a misspelled
`--format` is an error rather than a silent no-op that exits 0; and a swept or
pinned model no longer lands in `generationConfig.model`, where the Gemini API
does not read it.

## Documentation

- [Quickstart](docs/quickstart.md) — five minutes, no endpoint required
- [Why no composite score](docs/concepts/no-composite-score.md) — the design
  decision most likely to surprise you
- [Domination and paired tests](docs/concepts/domination.md) — how the
  statistics work
- [The judge](docs/concepts/judge.md) — what to do about responses no rule
  can score
- [Gating in CI](docs/guides/ci-gate.md) · [Safety](docs/guides/safety.md) ·
  [Plugin cookbook](docs/guides/plugins.md)
- [The OpenAI scorecard](docs/scorecard.md) — ten models, and what is not measurable yet
- [A committed example run](examples/README.md) you can re-report offline
- [Design specification](docs/superpowers/specs/2026-09-12-sweepeval-design.md)
  · [Implementation plan](docs/superpowers/plans/2026-09-12-sweepeval-v0.1-plan.md)
- [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## Licence

Apache-2.0.
