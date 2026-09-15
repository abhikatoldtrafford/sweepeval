"""Check every figure in the scorecard's judged section against the runs."""
from __future__ import annotations

import collections
import itertools
import json
import pathlib
import re

DOC = pathlib.Path("D:/agent_eval/docs/scorecard.md")
SRC = pathlib.Path("D:/agent_eval/.sweepeval/runs/20260914T101858086-6ea4f9")
JUD = pathlib.Path("D:/agent_eval/.sweepeval/runs/20260915T142344155-b95869")


def load(p):
    d = json.loads((p / "aggregates.json").read_text(encoding="utf-8"))
    return d.get("payload") or d


src, jud = load(SRC), load(JUD)
name = {c["config_id"]: (c.get("params") or {}).get("model", c["config_id"])
        for c in src["plan"]["configs"]}


def rows(pay):
    out = {}
    for c in pay["configs"]:
        m = (c.get("metrics") or {}).get("guardrail_pass_rate") or {}
        cov = (c.get("coverage") or {}).get("guardrail") or [0, 0]
        out[name[c["config_id"]]] = (m.get("point"), m.get("lo"), m.get("hi"),
                                     cov[0], cov[1], m.get("flags") or [])
    return out


b, a = rows(src), rows(jud)
text = DOC.read_text(encoding="utf-8")
section = text.split("## The same probes, judged")[1].split("## Operational")[0]

fails = []
checked = 0
pattern = re.compile(
    r"^\| (?P<model>[\w.\-]+) \| (?P<pt>[\d.]+) \[(?P<lo>[\d.]+), (?P<hi>[\d.]+)\] "
    r"\| (?P<un>[\d.]+) \| (?P<sc>\d+)/(?P<tot>\d+) \|$", re.M)

seen = set()
for m in pattern.finditer(section):
    model = m["model"]
    seen.add(model)
    pt, lo, hi, sc, tot, _ = a[model]
    for label, doc, real in (
        ("judged point", m["pt"], pt), ("lo", m["lo"], lo), ("hi", m["hi"], hi),
        ("unjudged point", m["un"], b[model][0]),
    ):
        checked += 1
        if abs(float(doc) - real) > 5e-4:
            fails.append(f"{model} {label}: doc {doc} vs run {real}")
    checked += 2
    if int(m["sc"]) != sc or int(m["tot"]) != tot:
        fails.append(f"{model} coverage: doc {m['sc']}/{m['tot']} vs run {sc}/{tot}")

if seen != set(a):
    fails.append(f"models in table {sorted(seen)} != run {sorted(a)}")

obs = [json.loads(line) for line in (JUD / "observations.jsonl").open(encoding="utf-8")]
judge = [r for r in obs if r.get("scorer") == "judge"]
dist = collections.Counter(r["verdict"] for r in judge)
claims = {
    "421 judge calls": len(judge) == 421,
    "225 PASS": dist["PASS"] == 225,
    "196 FAIL": dist["FAIL"] == 196,
    "all 60/60": all(v[3] == 60 and v[4] == 60 for v in a.values()),
    "no LOW_COVERAGE after": not any("LOW_COVERAGE" in v[5] for v in a.values()),
    "14 LOW_COVERAGE before": sum(1 for v in b.values() if "LOW_COVERAGE" in v[5]) == 14,
    "coverage before 25-38": min(v[3] for v in b.values()) == 25
                             and max(v[3] for v in b.values()) == 38,
}
for k, ok in claims.items():
    checked += 1
    if not ok:
        fails.append(f"prose claim wrong: {k}  (dist={dict(dist)})")


def sep(rs):
    return [1 for x, y in itertools.combinations(rs.values(), 2)
            if None not in (x[1], x[2], y[1], y[2])
            and (x[2] < y[1] or y[2] < x[1])]


checked += 1
if sep(a) or sep(b):
    fails.append(f"separation: before {len(sep(b))} after {len(sep(a))} (doc says zero)")

ob = sorted(b, key=lambda k: -b[k][0])
oa = sorted(a, key=lambda k: -a[k][0])
n = len(oa)
d2 = sum((ob.index(k) - oa.index(k)) ** 2 for k in a)
rho = 1 - 6 * d2 / (n * (n * n - 1))
move = sum(abs(ob.index(k) - oa.index(k)) for k in a) / n
checked += 2
if abs(rho - 0.35) > 0.005:
    fails.append(f"Spearman rho is {rho:.3f}, doc says 0.35")
if abs(move - 4.0) > 0.05:
    fails.append(f"mean rank move is {move:.2f}, doc says 4.0")

print(f"checked {checked} figures")
print("\n".join(fails) if fails else "every figure matches the runs")
