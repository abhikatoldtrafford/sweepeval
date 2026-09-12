"""HTML and JUnit reports for a sweep (spec §15, D30).

The HTML page renders **offline** with no network, no JavaScript and no
bundled library. Trade-off plots are inline SVG for the same reason: a report
a user opens on a laptop with no internet, or emails to a colleague, has to
draw itself. ``matplotlib`` stays an optional extra for people who want
publication figures; nothing here needs it.

Pairwise 2D views are the primary read (§14.6) — security x latency, cost x
guardrail, determinism x retention — with the full table below them. Six
dimensions do not fit on a page, and a six-column table is not a trade-off
plot however carefully it is formatted.

JUnit exists for one job: making a sweep legible in a CI run's test tab. Each
constraint violation and each confirmed hard fail is a failing case with the
number that failed in the message; everything else is a passing case carrying
its interval, so the CI summary shows what was measured rather than only what
broke.
"""

from __future__ import annotations

from html import escape
from typing import Any
from xml.etree import ElementTree as ET

__all__ = ["PAIRWISE_VIEWS", "as_html", "as_junit"]

PAIRWISE_VIEWS: tuple[tuple[str, str], ...] = (
    ("security_pass_rate", "latency_p95_ms"),
    ("cost_per_probe", "guardrail_pass_rate"),
    ("target_determinism_at_temp0", "context_retention_auc"),
)
"""§14.6's three primary views."""

_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif;
       margin: 0 auto; max-width: 60rem; padding: 2rem 1rem; }
h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #8884;
     padding-bottom: 0.25rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid #8883; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.dim { color: #8889; }
.warn { color: #b26b00; }
.bad { color: #c0392b; }
.plots { display: flex; flex-wrap: wrap; gap: 1rem; }
figure { margin: 0; }
figcaption { font-size: 0.85rem; color: #8889; }
.note { background: #8881; border-left: 3px solid #8884; padding: 0.6rem 0.9rem;
        margin: 0.8rem 0; }
"""


def as_html(sweep: Any, frontier: Any = None) -> str:
    """A self-contained page. No script tag, no external request."""
    labels = {row.config_id: row.config.label() for row in sweep.configs}
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        f"<title>sweepeval {escape(sweep.run_id)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>sweepeval — {escape(sweep.profile)} sweep</h1>",
        f'<p class="dim">run {escape(sweep.run_id)} · '
        f"{len(sweep.configs)} configuration(s) · {sweep.runs} runs · "
        f"status {escape(sweep.status.value)}</p>",
    ]

    if sweep.stop_reason:
        parts.append(f'<div class="note warn">{escape(sweep.stop_reason)}</div>')
    if sweep.profile == "quick":
        parts.append(
            '<div class="note">Intervals at <code>quick</code> are valid but '
            "wide, so few pairs will separate. These results are "
            "<strong>not gate-eligible</strong> — use "
            "<code>--profile standard</code> for decisions or gating.</div>"
        )

    if frontier is not None:
        parts.append(_plots(frontier, labels))
        parts.append(_frontier_section(frontier, labels))
    parts.append(_measurements(sweep))
    parts.append(_skipped(sweep))
    parts.append("</body></html>")
    return "\n".join(p for p in parts if p)


def _plots(frontier: Any, labels: dict[str, str]) -> str:
    available = {o.id for o in frontier.objectives}
    views = [v for v in PAIRWISE_VIEWS if v[0] in available and v[1] in available]
    if not views:
        return ""
    figures = "".join(_svg(frontier, labels, x, y) for x, y in views)
    return f"<h2>trade-offs</h2><div class='plots'>{figures}</div>"


def _svg(frontier: Any, labels: dict[str, str], x_id: str, y_id: str) -> str:
    """One 2D scatter, drawn by hand. Frontier members are filled."""
    width, height, pad = 260, 200, 34
    points = [
        (c, p[x_id], p[y_id])
        for c, p in sorted(frontier.points.items())
        if x_id in p and y_id in p
    ]
    if len(points) < 2:
        return ""

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    # A degenerate axis would divide by zero; widen it so the points land in
    # the middle rather than on the frame.
    if x_hi == x_lo:
        x_lo, x_hi = x_lo - 0.5, x_hi + 0.5
    if y_hi == y_lo:
        y_lo, y_hi = y_lo - 0.5, y_hi + 0.5

    def px(value: float) -> float:
        return pad + (value - x_lo) / (x_hi - x_lo) * (width - 2 * pad)

    def py(value: float) -> float:
        return height - pad - (value - y_lo) / (y_hi - y_lo) * (height - 2 * pad)

    marks = []
    for config_id, x_value, y_value in points:
        on_frontier = config_id in frontier.frontier
        fill = "currentColor" if on_frontier else "none"
        marks.append(
            f'<circle cx="{px(x_value):.1f}" cy="{py(y_value):.1f}" r="4" '
            f'fill="{fill}" stroke="currentColor" stroke-width="1.5">'
            f"<title>{escape(config_id)} {escape(labels.get(config_id, ''))}\n"
            f"{escape(x_id)}={x_value:.4g}\n{escape(y_id)}={y_value:.4g}</title>"
            "</circle>"
        )

    return (
        "<figure>"
        f'<svg width="{width}" height="{height}" role="img" '
        f'aria-label="{escape(x_id)} against {escape(y_id)}">'
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" '
        f'y2="{height - pad}" stroke="currentColor" opacity="0.4"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" '
        'stroke="currentColor" opacity="0.4"/>'
        + "".join(marks)
        + f'<text x="{width / 2}" y="{height - 6}" font-size="10" '
        f'text-anchor="middle" opacity="0.7">{escape(x_id)}</text>'
        f'<text x="10" y="{height / 2}" font-size="10" opacity="0.7" '
        f'transform="rotate(-90 10 {height / 2})">{escape(y_id)}</text>'
        "</svg>"
        f"<figcaption>filled = on the frontier</figcaption></figure>"
    )


def _frontier_section(frontier: Any, labels: dict[str, str]) -> str:
    rows = []
    for index, cluster in enumerate(frontier.clusters, start=1):
        members = "<br>".join(
            f"{escape(m)} <span class='dim'>{escape(labels.get(m, ''))}</span>"
            for m in cluster.members
        )
        spread = "<br>".join(
            f"{escape(k)}: {lo:.4g} .. {hi:.4g}"
            + (" <span class='warn'>wide</span>" if k in cluster.wide else "")
            for k, (lo, hi) in sorted(cluster.spread.items())
        )
        rows.append(f"<tr><td>cluster {index}</td><td>{members}</td><td>{spread}</td></tr>")

    out = [
        "<h2>frontier</h2>",
        f"<p class='dim'>{len(frontier.frontier)} non-dominated configuration(s) in "
        f"{len(frontier.clusters)} tied cluster(s), alpha {frontier.alpha:g} "
        "family-wise.</p>",
        "<table><thead><tr><th></th><th>members</th><th>spread</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>",
    ]

    if frontier.dominated:
        items = "".join(
            f"<li>{escape(c)} — dominated by {escape(', '.join(d))}</li>"
            for c, d in sorted(frontier.dominated.items())
        )
        out.append(f"<h2>dominated</h2><ul>{items}</ul>")

    if frontier.constraints is not None and frontier.constraints.violations:
        items = "".join(
            f"<li class='bad'>{escape(v.describe())}</li>"
            for v in frontier.constraints.violations
        )
        out.append(f"<h2>excluded by a hard constraint</h2><ul>{items}</ul>")

    for a, b, value in frontier.correlation.high_pairs():
        out.append(
            f"<div class='note warn'>{escape(a)} and {escape(b)} correlate at "
            f"{value:+.2f} — close to measuring one thing. Drop one with "
            "<code>--objectives</code>.</div>"
        )

    return "".join(out)


def _measurements(sweep: Any) -> str:
    from sweepeval.schema.metric import Flag

    names: list[str] = []
    for row in sweep.configs:
        for metric in row.metrics:
            if metric not in names:
                names.append(metric)
    names.sort()
    if not names:
        return ""

    header = "".join(f"<th class='num'>{escape(n)}</th>" for n in names)
    body = []
    for row in sweep.configs:
        cells = [f"<td>{escape(row.config.label())}</td>"]
        for name in names:
            value = row.metrics.get(name)
            if value is None:
                cells.append("<td class='num dim'>—</td>")
                continue
            if Flag.NO_VALID_INTERVAL in value.flags:
                cells.append("<td class='num dim'>—</td>")
                continue
            flags = " ".join(f.value for f in value.flags)
            cells.append(
                f"<td class='num'>{value.point:.4g}"
                f"<br><span class='dim'>[{value.lo:.3g}, {value.hi:.3g}]</span>"
                + (f"<br><span class='warn'>{escape(flags)}</span>" if flags else "")
                + "</td>"
            )
        body.append(f"<tr>{''.join(cells)}</tr>")

    return (
        "<h2>measurements</h2>"
        f"<table><thead><tr><th>config</th>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _skipped(sweep: Any) -> str:
    if not sweep.skipped:
        return ""
    items = "".join(
        f"<li><strong>{escape(f)}</strong> — {escape(r)}</li>"
        for f, r in sorted(sweep.skipped)
    )
    return f"<h2>SKIPPED</h2><ul>{items}</ul>"


def as_junit(sweep: Any, frontier: Any = None) -> str:
    """A JUnit XML document for a CI run's test tab."""
    from sweepeval.schema.metric import Flag

    suite = ET.Element(
        "testsuite",
        name="sweepeval",
        tests="0",
        failures="0",
        errors="0",
        skipped="0",
    )
    tests = failures = skipped = 0

    violations = (
        {v.config_id: v for v in frontier.constraints.violations}
        if frontier is not None and frontier.constraints is not None
        else {}
    )

    for row in sweep.configs:
        for hard in row.hard_fails.confirmed:
            tests += 1
            failures += 1
            case = ET.SubElement(
                suite,
                "testcase",
                classname=f"security.{row.config_id}",
                name=hard.unit_id,
            )
            ET.SubElement(case, "failure", message=hard.describe()).text = hard.reason

        violation = violations.get(row.config_id)
        if violation is not None:
            tests += 1
            failures += 1
            case = ET.SubElement(
                suite,
                "testcase",
                classname=f"constraint.{row.config_id}",
                name=violation.constraint.metric,
            )
            ET.SubElement(case, "failure", message=violation.describe())

        for name, value in sorted(row.metrics.items()):
            tests += 1
            case = ET.SubElement(
                suite, "testcase", classname=f"metric.{row.config_id}", name=name
            )
            if Flag.NO_VALID_INTERVAL in value.flags:
                # Not a pass: a metric nothing could be scored for is a
                # finding, and a green case would hide it (I5).
                skipped += 1
                ET.SubElement(
                    case,
                    "skipped",
                    message="no valid interval: too few scored clusters",
                )
            else:
                ET.SubElement(case, "system-out").text = (
                    f"{value.point:.6g} [{value.lo:.6g}, {value.hi:.6g}] "
                    f"n={value.n_clusters} "
                    + " ".join(f.value for f in value.flags)
                ).strip()

    for config_id in sweep.not_run:
        tests += 1
        skipped += 1
        case = ET.SubElement(
            suite, "testcase", classname="sweep", name=f"config {config_id}"
        )
        ET.SubElement(
            case, "skipped", message="never ran: the sweep stopped at its budget cap"
        )

    suite.set("tests", str(tests))
    suite.set("failures", str(failures))
    suite.set("skipped", str(skipped))
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(
        suite, encoding="unicode"
    )
