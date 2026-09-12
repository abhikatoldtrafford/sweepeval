# sweepeval

**Zero-config, black-box sweep and benchmark engine for LLM and agent systems.**

```bash
uvx sweepeval run https://your-endpoint --key $KEY
```

Point it at an HTTP endpoint. It works out the request shape, brings its own
probes, sweeps the configurations it can prove are variable, and returns a
Pareto frontier with a confidence interval on every number.

No config file. No prompts to write. No labels. No axes to declare.

## Start here

- **[Quickstart](quickstart.md)** — five minutes, no endpoint required.
- **[Why no composite score](concepts/no-composite-score.md)** — the design
  decision most likely to surprise you.
- **[Gating in CI](guides/ci-gate.md)** — a regression gate that does not flap.

## Try it with nothing

```bash
uvx sweepeval demo
```

Runs the whole pipeline — discovery, capability detection, the
sampling-effect test, a real sweep, real statistics — against a simulated
endpoint bundled in the package. Nothing is sent anywhere and nothing is
spent. The numbers describe a scripted mock; the *shape* is exactly what a
real run produces.

## The four claims

**Blind discovery.** Other tools need you to describe your endpoint and write
your evals. This one probes the endpoint to work out its request shape, infers
how to extract the response, and detects what it can actually do. If
temperature has no measurable effect on your endpoint, sweeping it is theatre,
and sweepeval says so instead of drawing a chart.

**Confidence intervals on everything, including the gate.** Gates on point
estimates flap, and flapping gates get disabled.

**No composite score.** Anywhere. See
[why](concepts/no-composite-score.md).

**Black box, always.** No SDK, no instrumentation, no framework hooks. Every
number is measured from what came back over HTTP.
