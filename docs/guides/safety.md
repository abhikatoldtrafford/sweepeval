# Safety and authorization

## What this tool sends

sweepeval's security suite is an **active prompt-injection and exfiltration
test suite**. Pointing it at an endpoint sends live requests containing
adversarial prompts. On an agent system those requests may trigger real tool
calls, real writes, and real spend.

Only run it against endpoints you are authorised to test.

## The authorization affirmation

Before the security family runs against anything that is not localhost, the
tool requires a one-time per-host affirmation:

```bash
sweepeval sweep https://your-endpoint --key $KEY --i-am-authorized
```

Interactively, it prompts. The affirmation is recorded in the run's manifest
with a timestamp and remembered per host, so you affirm once rather than
every run. It is a deliberate speed bump, not a legal instrument: it exists so
that pointing an injection suite at a stranger's endpoint requires a
conscious act.

## What discovery sends

Discovery is deliberately inert. It sends one fixed prompt —
`Reply with the single word OK.` — and nothing else. A test greps every
discovery request body for it, because on an agent target a "read-shaped"
body is still a live prompt that can trigger a tool call.

Hard limits, none of them adjustable upward by accident:

- at most **25** requests;
- a wall-clock cap;
- a token cap;
- full-jitter exponential backoff, with a circuit breaker after five
  consecutive failures;
- per-target concurrency of 2 by default.

Never an indefinite loop against an unknown endpoint. When the budget is spent
without identifying a shape, the tool aborts and prints the full transcript —
every shape, every mutation, every status and error body — because that
transcript is the deliverable when discovery fails.

## What the probes contain

The corpus ships in full, in YAML, framed as a test suite for a fictional
supply company. It is published rather than obfuscated: a probe suite you
cannot read is a probe suite you cannot audit, and the attacks in it are all
publicly documented classes mapped to OWASP LLM Top 10 categories.

Harm-enablement probes stop at the **refusal boundary**. They test whether the
target declines; they never need to contain operational content, and they do
not.

## Credentials

Your key is redacted from every artifact: request bodies, URLs, error
excerpts, manifests, and reports. Redaction happens in the store, and the
store is the only way to obtain a writer, which is what makes the no-secrets
contract test exhaustive rather than a spot check — there is no other
constructor path to reach past.

`baseline.json` is designed to be committed and carries no credentials.

```bash
sweepeval init
```

writes `.gitignore` entries for `.sweepeval/` and prints exactly what is and
is not safe to commit.

For a sensitive target, `--no-store-bodies` disables response-text storage. It
disables semantic stability and judge escalation as a consequence, and says
so rather than silently degrading them.

## Telemetry

None. No analytics, no phone-home, no crash reporting, no version check. The
only network calls the tool makes are to the target you name.

## Reporting a security issue

If you find a way to make the tool send something it should not — a
destructive discovery request, an unredacted credential in an artifact, the
security suite running without the affirmation — please
[report it privately](https://github.com/abhikatoldtrafford/sweepeval/security/advisories/new)
rather than opening an issue.
