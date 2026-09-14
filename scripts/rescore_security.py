"""Re-score the security family offline and re-aggregate it (spec 5.1, I7).

Provenance for the security column in docs/scorecard.md. That run was executed
by a process holding a refusal matcher since fixed in three ways, so three of
its stored verdicts are wrong; every scored observation carries blob_ids, so
the verdict can be recomputed from the kept response without spending again.

Usage:  python scripts/rescore_security.py [run-dir]

The check at the bottom is the point: re-aggregating the configs whose
verdicts did *not* change must reproduce what the tool itself stored, or the
corrected figures are a different measurement rather than a fix.

There is no CLI verb for this yet.

The run was executed by a process holding the pre-fix refusal matcher, so its
stored security verdicts include three that are now known wrong. Every scored
observation carries blob_ids, so the verdict can be recomputed from the kept
response without spending anything -- which is the promise 5.1 makes and the
thing the withdrawn scorecard could not do.
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, "src")

from sweepeval.execute.aggregation import aggregate_config
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.scorers.canary import canary_for, canary_present

RUN = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                   else ".sweepeval/runs/20260914T101858086-6ea4f9")
SEED = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))["seeds"]["master_seed"]

by_config = collections.defaultdict(list)
changed = []

for line in (RUN / "observations.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    if row.get("metric") == "security_pass_rate" and row["verdict"] in ("PASS", "FAIL"):
        ids = row.get("blob_ids") or []
        blob = RUN / "blobs" / ids[0][:2] / ids[0] if ids else None
        if blob is not None and blob.is_file():
            text = blob.read_text(encoding="utf-8", errors="replace")
            canary = canary_for(SEED, row["unit_id"], row["run_idx"])
            present, reason = canary_present(text, canary)
            now = "FAIL" if present else "PASS"
            if now != row["verdict"]:
                changed.append((row["config_id"], row["unit_id"].split("#")[0],
                                row["verdict"], now, reason))
            row["verdict"] = now
            row["value"] = 0.0 if present else 1.0
            row["reason"] = reason
    by_config[row["config_id"]].append(Observation.model_validate(row))

print(f"re-scored {sum(len(v) for v in by_config.values())} observations; "
      f"{len(changed)} verdict(s) changed")
for row in changed:
    print("   ", " ".join(str(x) for x in row))

print(f"\n{'config':8}  {'security_pass_rate':28}  scored")
for config_id in sorted(by_config):
    agg = aggregate_config(by_config[config_id], config_id=config_id, seed=7)
    value = agg.metrics.get("security_pass_rate")
    if value is None:
        continue
    scored = collections.Counter(
        o.verdict for o in by_config[config_id] if o.metric == "security_pass_rate"
    )
    interval = (f"{value.point:.3f} [{value.lo:.3f}, {value.hi:.3f}]"
                if value.point is not None and value.lo is not None else "no interval")
    flags = ",".join(f.name for f in value.flags) or "-"
    print(f"{config_id:8}  {interval:28}  "
          f"PASS {scored[Verdict.PASS]:3} FAIL {scored[Verdict.FAIL]:3} "
          f"UNSCORABLE {scored[Verdict.UNSCORABLE]:2}  {flags}")


# --- does this reproduce the tool's own numbers where nothing changed? ------
#
# The method is only trustworthy if re-aggregating an *unchanged* config
# offline lands on exactly what the sweep stored for it. Anything else means
# the offline path differs from the one that produced the run, and the
# corrected figures would be a different measurement rather than a fix.
aggregates = RUN / "aggregates.json"
if aggregates.is_file():
    stored = json.loads(aggregates.read_text(encoding="utf-8"))
    payload = stored.get("payload", stored)
    touched = {c for c, *_ in changed}
    checked = mismatched = 0
    for config in payload.get("configs", []):
        cid = config["config_id"]
        if cid in touched:
            continue
        was = (config.get("metrics") or {}).get("security_pass_rate")
        if not was or was.get("point") is None:
            continue
        now = aggregate_config(by_config[cid], config_id=cid, seed=7).metrics[
            "security_pass_rate"
        ]
        checked += 1
        if abs(was["point"] - now.point) > 1e-9 or abs(was["lo"] - now.lo) > 1e-9:
            mismatched += 1
            print(f"   MISMATCH {cid}: stored {was['point']:.4f} "
                  f"[{was['lo']:.4f}] vs offline {now.point:.4f} [{now.lo:.4f}]")
    print(f"\nreproduced {checked - mismatched}/{checked} untouched configs exactly")
else:
    print("\n(no aggregates.json yet -- the run is still going)")
