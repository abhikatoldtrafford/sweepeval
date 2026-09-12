# Domination and paired tests

## The pairing

Every configuration in a sweep faces the **identical probe set** with
**identical canary values**. That is invariant I4, and it is checkable from
`plan.json`: the units and the canary table are written once, outside the
config list, before the first scoring request.

That sharing is not an optimisation. It makes the probe a *matched block*, so
the difference between two configurations can be taken within it. A probe that
is hard for one configuration is hard for the other, and the difference does
not care how hard it was. This is where the statistical power at N=3 comes
from.

## The test

For each objective, a **paired cluster bootstrap** resamples the clusters —
the probe, or for retention the conversation — and computes the difference
`b - a` on each replicate, oriented so positive always means `b` is ahead.

Each objective resamples *its own* cluster set. Retention resamples within
depth, because an unstratified resample empties a depth in a few percent of
replicates and the trapezoid is undefined there.

Two one-sided p-values come out of the same replicate distribution, against
thresholds shifted by the objective's minimum effect:

- **superiority** — `b` is better by at least the margin;
- **non-inferiority** — `b` is not worse by more than the margin.

Non-inferiority is a real test, not a failure to detect a difference.
Establishing "not worse" from "we did not find a difference" is absence of
evidence, and this tool refuses that reasoning consistently.

## The rule

`b` **dominates** `a` when `b` is non-inferior on **every** objective and
superior on **at least one**.

- The non-inferiority half is an intersection-union test: the pair's p-value
  is the *maximum* of the per-objective non-inferiority p-values, with no
  correction, because rejecting the union requires rejecting every component.
- The superiority half is a union: Bonferroni across objectives.
- The pair's p-value is the maximum of the two halves.
- **Holm** then corrects across every ordered pair in the sweep.

Domination is asserted only when Holm rejects at the stated family-wise error
rate. That is invariant I2, and only one module — `rank.domination` — is
allowed to make the call. The layering contract enforces it, because a
domination decided anywhere else would not go through Holm.

## Conservative by design

The rule cannot discard a configuration that would have reached the frontier.
Measured on simulation: false domination **0.000** at 3, 5 and 8
configurations; power **0.985** on a genuine separation.

The cost is that trade-off pairs stay on the frontier, and at `quick` most
pairs do. That is the intended behaviour: a frontier of everything is an
honest statement that the data does not separate them.

## Tied clusters

Configurations that no test separates are grouped by **complete linkage** cut
exactly at the tie boundary: a cluster may contain a set of configurations
only if *every* pair within it is tied.

Connected components would be the obvious implementation and are wrong. The
tie relation is not transitive — A ties B, B ties C, and A is significantly
better than C is an ordinary outcome — so components chain, and with six
objectives their modal outcome is one cluster containing everything.

Because every internal pair is tied by construction, a cluster's diameter is
zero and reporting it would be vacuous. What is reported instead is
**spread**: the observed range of each objective across the members, with a
note where that range is wide despite the pairs being indistinguishable. Wide
spread with no significance is a signal to raise N, and the report says so.

The cover is not unique, so the tie-break is fixed and disclosed: fewest
clusters, then lexicographically smallest by sorted member ids. A reader
comparing two runs needs the grouping to be stable.

## Coverage parity

Before any of this, per-family scored counts are compared across
configurations. Divergence above 10% **blocks domination for that pair**: the
paired test pairs on the probe, and a probe only one side scored is not a
matched block. Above 30% the metric is flagged `LOW_COVERAGE` and excluded.

Blocked pairs stay in the Holm family rather than being removed from it.
Removing them would shrink the family and loosen every other pair's threshold,
which can only ever create a domination the full family would not have
licensed.

## The determinism confound

`target_determinism_at_temp0` is measured at a pinned `temp=0` and **shared**
across a (model, system_prompt) pair's temperature siblings. Measured at each
configuration's own settings it would be ~1.0 for every `temp=0`
configuration and ~0 for every `temp=1.0` one *by construction* — a
manufactured win on a metric that merely restates the row's own label, leaving
every `temp=0` configuration non-dominated for free.

Sharing relocates the structure rather than removing it: the objective takes
one value per pair and is exactly tied within each temperature triple. A tie
does not block domination the way a manufactured win does.

`config_repeatability` — the byte-match rate at the configuration's *own*
settings — is reported beside it as the production-truth figure, and is
promotable onto the frontier with `--objectives config_repeatability` if that
is the number you want, confound included.
