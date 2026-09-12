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
+--------+------------+------------+------------+------------+------------+
| config |   security |  guardrail |  retention |  det@temp0 | latency ms |
+--------+------------+------------+------------+------------+------------+
| cfg-00 |  1 [0.679, |  1 [0.596, |  1 [0.679, |  1 [0.679, |     0.7775 |
|        |         1] |   1] LOW_N |         1] |         1] |    [0.748, |
|        |            |            |            |            |     0.807] |
| cfg-01 |  1 [0.679, |  1 [0.596, |  1 [0.679, |  1 [0.679, |     0.8877 |
|        |         1] |   1] LOW_N |         1] |         1] |    [0.837, |
|        |            |            |            |            |     0.941] |
+--------+------------+------------+------------+------------+------------+

coverage (scored / attempted)
  cfg-00     context 30/30  determinism 30/30  guardrail 21/30  security 30/30

SKIPPED
  retrieval        retrieval=UNSUPPORTED (citation probe)
  tool_integrity   tool_calling=UNSUPPORTED (tool probe)
  degradation      not_implemented_in_v0.1: concurrency ramp, long inputs and
                   induced tool failures land in v0.2

frontier - 4 non-dominated config(s) in 1 tied cluster(s), at alpha 0.05 family-wise

  cluster 1 (4 config(s))
    cfg-00  temperature=0.0 sys=none
    cfg-01  temperature=1.0 sys=none
    cfg-03  temperature=1.0 sys=terse_neutral
    cfg-04  temperature=0.0 sys=verbose_strict_with_guardrails
    spread across members:
      latency_p95_ms                   0.7775 .. 0.9133 wide
    the range above is wide although no pair separates statistically -
    raise --runs, or use --profile standard

dominated
  cfg-02  temperature=0.0 sys=terse_neutral
    dominated by cfg-00
    cfg-00 is better on: latency_p95_ms
    cfg-00 gives up:     nothing
    p=0 against a Holm threshold of 0.001667
```

Every number carries an interval. Nothing carries a rank. Every family the
endpoint could not support says which detector ruled it out.

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
  --prefer "maximize security_pass_rate subject to latency_p95_ms < 2000"

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

## Cost and wall-clock

Every run prints its estimate and asks before spending. The estimate covers
discovery and capability detection too, not just scoring — those spend first,
and gating after them would be gating after the money was gone.

| Profile | Units | Calls/run | Configs | Requests at 3 runs | Wall-clock at concurrency 2 |
|---|---|---|---|---|---|
| `quick` | 40 | 62 | 6 | 1,201 | ~25 min |
| `standard` | 80 | 191 | 12 | 6,961 | ~145 min |

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
itself needs, nothing is sent at all.

## In CI

```yaml
- run: pipx install sweepeval
- run: |
    sweepeval sweep ${{ vars.ENDPOINT }} --key ${{ secrets.KEY }} \
      --profile standard --yes --format junit
    sweepeval gate ${{ vars.ENDPOINT }} --key ${{ secrets.KEY }} \
      --baseline .sweepeval/baseline.json
```

`gate` exits `0` unchanged, `1` regressed, `2` incomparable, `3` inconclusive.
It compares against a committed `baseline.json` using the same paired test,
so it does not flap.

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
- [Gating in CI](docs/guides/ci-gate.md) · [Safety](docs/guides/safety.md) ·
  [Plugin cookbook](docs/guides/plugins.md)
- [A committed example run](examples/README.md) you can re-report offline
- [Design specification](docs/superpowers/specs/2026-09-12-sweepeval-design.md)
  · [Implementation plan](docs/superpowers/plans/2026-09-12-sweepeval-v0.1-plan.md)
- [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md)

## Licence

Apache-2.0.
