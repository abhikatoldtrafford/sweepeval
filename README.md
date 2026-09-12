# sweepeval

**Zero-config, black-box sweep and benchmark engine for LLM and agent systems.**

Point it at an HTTP endpoint. It discovers the endpoint's shape, brings its own
probes, sweeps the configurations it can prove are variable, and returns a Pareto
frontier with confidence intervals on every number.

```bash
uvx sweepeval run https://your-endpoint --key $KEY
```

No config file. No prompts to write. No labels. No axes to declare.

> **Status: pre-alpha, under construction.** Nothing works yet. The design is
> complete and audited (`docs/superpowers/specs/`), the implementation plan is
> written (`docs/superpowers/plans/`), and the code is being built against them
> milestone by milestone. Watch the repo if you want to know when it runs.

---

## What makes it different

**Blind discovery.** Every comparable tool needs you to describe your endpoint
and write your evals. This one probes the endpoint to work out its request
shape, infers how to extract the response, and detects what it can actually do —
streaming, tools, multi-turn state, whether temperature has any effect at all.
That last one matters: if temperature does nothing on your endpoint, sweeping it
is theatre, and sweepeval will tell you so instead of producing a chart.

**Confidence intervals on everything.** Including the regression gate. Eval tools
that report point estimates produce gates that flap, and flapping gates get
disabled. Every comparison here is a paired test over the identical probe set.

**No composite score.** A single number hides the trade-off that makes a
configuration decision hard: the most secure config is usually the slowest, and
the cheapest one usually leaks. sweepeval reports the frontier and says what each
position wins and gives up. Tell it your preference and it will name a config —
that preference is yours, applied at report time, never baked into the data.

**Black box, always.** No SDK coupling, no instrumentation, no framework hooks.
Every number is measured from what came back over HTTP.

## What it measures

| Family | What it checks |
|---|---|
| Security | Prompt injection, instruction override, system-prompt exfiltration, role confusion, delimiter escape, indirect and cross-turn injection |
| Guardrails | Policy adherence at four escalating pressure levels, reported per policy |
| Determinism | Exact repeatability, semantic stability, and invariance under paraphrase |
| Context | Fact recall as a decay curve, constraint persistence, contradiction handling, positional bias |
| Operational | First-token and total latency at p50/p95/p99, tokens, cost, error rate by class |

Anything the endpoint cannot support reports `SKIPPED` with a reason. Never a
silent pass, fail, or zero.

## Safety

sweepeval's security suite is an active prompt-injection and exfiltration test
suite. Pointing it at an endpoint sends live requests, and on an agent system
those requests may trigger real tool calls, writes, or spend.

**Only run it against endpoints you are authorised to test.** The tool requires a
one-time per-host affirmation before the security family runs against anything
that is not localhost.

No telemetry. No analytics. No network calls except to the target you name.

## Documentation

- [Design specification](docs/superpowers/specs/2026-09-12-sweepeval-design.md)
- [Implementation plan](docs/superpowers/plans/2026-09-12-sweepeval-v0.1-plan.md)

## Licence

Apache-2.0.
