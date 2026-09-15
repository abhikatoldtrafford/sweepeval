"""Is the judge reproducible? Two judged passes over identical stored text."""
from __future__ import annotations

import json
import pathlib

A = pathlib.Path("D:/agent_eval/.sweepeval/runs/20260915T140456431-24e628")
B = pathlib.Path("D:/agent_eval/.sweepeval/runs/20260915T142344155-b95869")


def verdicts(run):
    out = {}
    for line in (run / "observations.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("scorer") == "judge":
            out[(r["config_id"], r["unit_id"], r["run_idx"])] = r["verdict"]
    return out


a, b = verdicts(A), verdicts(B)
shared = set(a) & set(b)
disagree = [k for k in shared if a[k] != b[k]]

print(f"judge verdicts, pass A : {len(a)}")
print(f"judge verdicts, pass B : {len(b)}")
print(f"judged in both passes  : {len(shared)}")
print(f"disagreements          : {len(disagree)}"
      f"  ({100 * len(disagree) / max(len(shared), 1):.2f}%)")
print(f"only in B              : {len(set(b) - set(a))}")

for key in sorted(disagree)[:8]:
    print(f"   {key[0]} {key[1].split('#')[0]:30} run{key[2]}  "
          f"{a[key]} -> {b[key]}")
