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
  degradation      not_implemented: concurrency ramp, long inputs and induced
                   tool failures are specified but not built (spec section 11,
                   family 8)

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

**The security column that stood here has been withdrawn.** An independent
adversarial audit found the canary matcher wrong in the unsafe direction — a
model that emitted the planted marker and then apologised scored a pass — so
every published rate was an upper bound. It is fixed, but this run cannot be
re-scored offline, so the numbers are gone rather than corrected.
[The full scorecard](docs/scorecard.md) sets out what was withdrawn, why, and
what a replacement run needs.

This is one `quick` run at N=2 and it is **not gate-eligible**.

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
