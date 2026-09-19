"""Check every figure in docs/scorecard-families.md against the run.

Prose figures in this project have gone wrong twice by hand -- a coverage
ratio published as a call count, and a block of PASS/FAIL counts carried from
one judged pass into text about the next -- and neither was caught by reading.
So the document is parsed and every number in it is re-derived from
`observations.jsonl` and `aggregates.json`.

Exit 0 and a count of what was checked, or exit 1 naming each disagreement.
"""
from __future__ import annotations

import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "scorecard-families.md"
RUN = ROOT / ".sweepeval" / "runs" / "20260919T075816521-99b26e"

TOL = 5e-4  # the document quotes three decimals


def load():
    agg = json.loads((RUN / "aggregates.json").read_text(encoding="utf-8"))["payload"]
    obs = [json.loads(l) for l in
           (RUN / "observations.jsonl").read_text(encoding="utf-8").splitlines() if l]
    calls = [json.loads(l) for l in
             (RUN / "calls.jsonl").read_text(encoding="utf-8").splitlines() if l]
    man = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    return agg, obs, calls, man


agg, obs, calls, man = load()
text = DOC.read_text(encoding="utf-8")

model_of = {c["config_id"]: (c.get("params") or {}).get("model")
            for c in agg["configs"]}
by_model = {(c.get("params") or {}).get("model"): c for c in agg["configs"]}

fails: list[str] = []
checked = 0


def check(label: str, want, got) -> None:
    global checked
    checked += 1
    ok = (abs(want - got) <= TOL) if isinstance(want, float) else (want == got)
    if not ok:
        fails.append(f"{label}: document says {want!r}, run says {got!r}")


def stated(pattern: str, label: str, got, cast=int) -> None:
    """Assert the document contains a figure, and that it matches."""
    global checked
    m = re.search(pattern, text)
    if m is None:
        checked += 1
        fails.append(f"{label}: no figure matching {pattern!r} in the document")
        return
    check(label, cast(m.group(1).replace(",", "")), got)


# --- run-level totals --------------------------------------------------------

stated(r"\*\*([\d,]+) requests in 1h52m\*\*", "total requests", len(calls))
stated(r"Run id `([\dA-Za-z]+-[\da-f]+)`", "run id", agg["run_id"], cast=str)
stated(r"`corpus_hash`\n`([0-9a-f]{64})`", "corpus_hash", man["corpus_hash"], cast=str)

# --- capability gating table -------------------------------------------------

row = re.compile(
    r"^\| (?P<model>[\w.\-]+) \| (?P<ret>[A-Z]+) \| (?P<tool>[A-Z]+) \| "
    r"`(?P<skip>\w+)` \|$", re.M)
seen_models = set()
for m in row.finditer(text):
    model = m.group("model")
    seen_models.add(model)
    caps = by_model[model]["capabilities"]
    check(f"{model} retrieval verdict", m.group("ret"),
          caps["retrieval"]["verdict"])
    check(f"{model} tool_calling verdict", m.group("tool"),
          caps["tool_calling"]["verdict"])
    skipped = [f for f, _ in by_model[model]["skipped"]]
    check(f"{model} skipped family", [m.group("skip")], skipped)

if len(seen_models) != 4:
    fails.append(f"capability table covers {len(seen_models)} models, expected 4")
checked += 1

# --- verdict counts, from observations --------------------------------------

tally: dict = collections.defaultdict(collections.Counter)
for o in obs:
    tally[(model_of[o["config_id"]], o["metric"])][o["verdict"]] += 1

# "31 PASS, 5 FAIL" style claims, tied to the metric named in the same row
for metric, model, passes, failures in [
    ("citation_integrity", "gpt-5-search-api", 31, 5),
    ("citation_stability", "gpt-5-search-api", 2, 10),
]:
    counts = tally[(model, metric)]
    pattern = (rf"`{metric}` \| \*?\*?[\d.]+ \[[\d., ]+\]\*?\*? — "
               rf"(\d+) PASS, (\d+) FAIL")
    m = re.search(pattern, text)
    checked += 1
    if m is None:
        fails.append(f"{metric}: no PASS/FAIL figures found in the document")
    else:
        check(f"{metric} PASS", int(m.group(1)), counts["PASS"])
        check(f"{metric} FAIL", int(m.group(2)), counts["FAIL"])
    check(f"{metric} PASS (expected)", passes, counts["PASS"])
    check(f"{metric} FAIL (expected)", failures, counts["FAIL"])

# --- every interval quoted anywhere in the document --------------------------

interval = re.compile(r"(?P<pt>\d\.\d{3}) \[(?P<lo>\d\.\d{3}), (?P<hi>\d\.\d{3})\]")
quoted = collections.Counter()
for m in interval.finditer(text):
    quoted[(m.group("pt"), m.group("lo"), m.group("hi"))] += 1

available = set()
for model, cfg in by_model.items():
    for name, value in (cfg.get("metrics") or {}).items():
        if value.get("lo") is None:
            continue
        available.add((f"{value['point']:.3f}",
                       f"{value['lo']:.3f}", f"{value['hi']:.3f}"))

for triple, n in quoted.items():
    checked += 1
    if triple not in available:
        fails.append(
            f"interval {triple[0]} [{triple[1]}, {triple[2]}] appears in the "
            f"document {n}x but matches no metric in the run"
        )

# --- SKIPPED IR metrics ------------------------------------------------------

skipped_rows = sum(1 for o in obs if o["verdict"] == "SKIPPED")
stated(r"(\d+) SKIPPED rows", "SKIPPED rows", skipped_rows)

# --- hard fails --------------------------------------------------------------

stated(r"`gpt-4\.1-nano` confirmed \*\*(\d+) hard fails",
       "nano hard fails", len(by_model["gpt-4.1-nano"]["hard_fails"]))
stated(r"and `gpt-5-search-api` \*\*(\d+)\*\*",
       "search hard fails", len(by_model["gpt-5-search-api"]["hard_fails"]))
stated(r"\(plus (\d+)\s*\n?suspected\)",
       "search suspected", len(by_model["gpt-5-search-api"]["suspected_hard_fails"]))

# --- transport errors --------------------------------------------------------

bad = [c for c in calls if (c.get("response") or {}).get("status") == 0]
stated(r"\*\*(\d+) transport errors across (?:\d+) units\*\*",
       "transport errors", len(bad))
stated(r"transport errors across \*?\*?(\d+) units",
       "transport units", len({c["unit_id"] for c in bad}))

per_config = collections.Counter(model_of[c["config_id"]] for c in bad)
stated(r"(\d+) were in the `gpt-4o-mini` config",
       "transport in 4o-mini", per_config["gpt-4o-mini"])
stated(r"and (\d+) in `gpt-5\.4-mini`",
       "transport in 5.4-mini", per_config["gpt-5.4-mini"])
stated(r"\*\*(\d+) of the \d+ units recovered on retry",
       "units recovered", len({c["unit_id"] for c in bad}) - 1)

# gpt-4o-mini's context denominator, stated in the document as 35
ctx = tally[("gpt-4o-mini", "fact_recall")]
stated(r"context denominator is (\d+), not 36",
       "4o-mini context PASS", ctx["PASS"])

# --- cluster floor: nothing quoted below it ---------------------------------

FLOOR = 8
for model, cfg in by_model.items():
    for name, value in (cfg.get("metrics") or {}).items():
        n = value.get("n_clusters") or 0
        if value.get("lo") is None or n >= FLOOR:
            continue
        triple = (f"{value['point']:.3f}", f"{value['lo']:.3f}",
                  f"{value['hi']:.3f}")
        checked += 1
        if triple in quoted:
            fails.append(
                f"{model} {name} has n={n}, below the cluster floor of "
                f"{FLOOR}, but its interval is quoted"
            )

# --- report ------------------------------------------------------------------

if fails:
    print(f"{len(fails)} disagreement(s) between the document and the run:\n")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)

print(f"{checked} figures checked against {agg['run_id']}; all agree.")
