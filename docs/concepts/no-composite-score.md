# Why there is no composite score

sweepeval never computes one. Not in ranking, not in the gate, not in any
report, not in any stored artifact. There is no `rank` field, no `score`, no
"overall". A test walks `frontier.json` asserting none appears.

## The reason

A single number requires weights, and the weights are the answer. When a tool
reports "config A: 8.4, config B: 7.9", it has decided on your behalf how much
security is worth in units of latency. You cannot see that decision, you did
not make it, and it is almost certainly wrong for your system — the right
trade-off for a customer-support bot and for an internal code assistant are
not the same trade-off.

Worse, the number *hides the thing that makes the decision hard*. In practice
the most secure configuration is usually the slowest, and the cheapest one
usually leaks. That tension is the finding. Averaging it away produces a
ranking that looks decisive and is arbitrary.

## What you get instead

**The frontier.** Every configuration that nothing else beats outright, with
what each one wins and gives up. Configurations that cannot be separated
statistically are grouped into a tied cluster, and the cluster reports its
*spread* — the observed range of each objective across its members — so a
wide range with no significance reads as "raise N", not as "these are
interchangeable".

**Your preference, applied at report time.** `--prefer` takes a priority order
or a constrained optimisation, names one configuration, and prints both the
preference and what that configuration concedes. It is off by default, it is
never stored, and changing it needs no re-run — because it is applied to
stored aggregates, not baked into them.

```bash
sweepeval report .sweepeval/runs/<id> --prefer security,cost,latency
sweepeval report .sweepeval/runs/<id> \
  --prefer "minimize cost_per_probe subject to security_pass_rate > 0.9"
```

## "But the frontier said everything is tied"

That happens, particularly at `--profile quick`, and it is a true statement
about your data rather than a failure of the tool. Three things you can do,
in increasing order of cost:

1. `--prefer` — apply your own priority to what was measured. Free, offline.
2. `--objectives security,latency` — narrow to the dimensions you actually
   care about. Also free and offline: every configuration ran the full
   profile, so narrowing afterwards is safe.
3. `--profile standard` or more runs — narrower intervals, more separations.

The report tells you which of these applies. What it will not do is
manufacture a ranking out of numbers that do not support one.
