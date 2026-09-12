"""`sweepeval discover` (spec §4.1, §8).

Phase 0 only: probe the endpoint, emit the annotated config, print the
evidence. Discovery is a bootstrap the user can correct, so the terminal output
leads with what was inferred and how confident the tool is — not with a
success banner.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sweepeval.discovery.budget import DiscoveryAborted, DiscoveryBudget
from sweepeval.discovery.emit import emit_config
from sweepeval.discovery.runner import DiscoveryOutcome, discover_target

console = Console()

_CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red", "none": "red"}


def discover_command(
    url: str = typer.Argument(..., help="The endpoint to probe."),
    key: str | None = typer.Option(None, "--key", "-k", help="API key, if required."),
    root: Path = typer.Option(Path(".sweepeval"), "--root", help="Artifact root."),
    max_posts: int = typer.Option(25, "--max-posts", help="Hard POST budget (§8.3)."),
    force: bool = typer.Option(False, "--force", help="Overwrite a hand-edited config."),
    seed: int | None = typer.Option(None, "--seed"),
) -> None:
    """Probe an endpoint and emit an annotated, editable config."""
    import httpx

    async def run() -> DiscoveryOutcome:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            return await discover_target(
                client, url, key, budget=DiscoveryBudget(max_posts=max_posts), seed=seed
            )

    try:
        outcome = asyncio.run(run())
    except DiscoveryAborted as aborted:
        console.print(f"[red]{aborted.render()}[/red]")
        raise typer.Exit(code=3) from None

    written = emit_config(
        root, url, outcome.payload,
        fingerprint=outcome.payload["target"]["endpoint_fingerprint"],
        force=force,
    )
    _render(outcome, written.path)


def _render(outcome: DiscoveryOutcome, config_path: Path) -> None:
    target = outcome.payload["target"]
    extraction = outcome.payload["extraction"]

    table = Table(title="discovered", show_header=True, header_style="bold")
    table.add_column("what")
    table.add_column("value")
    table.add_column("how")
    table.add_column("confidence")

    table.add_row("shape", target["shape"], target["shape_method"], "high")
    table.add_row("auth", target["auth"], target["auth_method"], "high")
    table.add_row(
        "target type",
        target["target_type"],
        "structural evidence",
        f"[{_CONFIDENCE_STYLE[target['target_type_confidence']]}]"
        f"{target['target_type_confidence']}[/]",
    )
    table.add_row(
        "text path",
        str(extraction["text_path"]),
        extraction["method"],
        f"[{_CONFIDENCE_STYLE.get(extraction['confidence'], 'yellow')}]"
        f"{extraction['confidence']}[/]",
    )
    console.print(table)

    models = outcome.payload["discovery"]["models_seen"]
    console.print(f"models exposed: {', '.join(models) if models else '(none)'}")
    console.print(f"discovery cost: {outcome.budget.posts} POST(s)")

    assumptions = [
        row
        for row in (
            (
                "extraction path",
                extraction["confidence"],
                "correct `extraction.text_path` and re-run",
            )
            if extraction["confidence"] in {"low", "medium", "none"}
            else None,
            (
                "target type",
                target["target_type_confidence"],
                "correct `target.target_type`; it gates which results may be compared",
            )
            if target["target_type_confidence"] in {"low", "medium"}
            else None,
        )
        if row
    ]
    if assumptions:
        console.print("\n[yellow]assumptions you can correct:[/yellow]")
        for what, confidence, how in assumptions:
            console.print(f"  {what} ({confidence}) — {how}")

    console.print(f"\nwrote {config_path}")
