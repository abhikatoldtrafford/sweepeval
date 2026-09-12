"""``sweepeval demo`` (spec §20, D33).

No URL, no key, no spend. It runs the full pipeline — discovery, capability
detection, the sampling-effect test, a real sweep, real scoring, real
statistics — against a **simulated** endpoint that ships inside the package.

Every line of its output says so. The mock is a fixture for exercising the
tool, and presenting its numbers as evidence about anything would be the same
category of dishonesty the tool exists to avoid: a mechanism that looks like a
measurement and is not.

What the demo is honest about:

* the target is simulated, and the banner repeats it;
* the numbers describe the mock's scripted behaviour, not any real model;
* the *shape* of the output — axes, frontier, intervals, SKIPPED reasons — is
  exactly what a real run produces, which is the thing worth showing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from sweepeval.cli.sweep import _finish

console = Console()

DEMO_URL = "https://demo.sweepeval.invalid/v1/chat/completions"
"""An ``.invalid`` host, so a demo that somehow escaped the in-process
transport would fail to resolve rather than reach a stranger's endpoint."""


def demo_command(
    root: Path = typer.Option(Path(".sweepeval"), "--root"),
    profile: str = typer.Option("quick", "--profile"),
    runs: int = typer.Option(2, "-n", "--runs"),
    prefer: str | None = typer.Option(None, "--prefer"),
    scenario: str = typer.Option("demo", "--scenario", help="A bundled scenario."),
    keep: bool = typer.Option(
        False, "--keep", help="Keep the artifacts instead of using a temp directory."
    ),
) -> None:
    """Run the whole pipeline against a bundled, simulated endpoint."""
    console.print(
        Panel(
            "[bold]This is a demo against a simulated endpoint.[/bold]\n"
            "Nothing is sent anywhere, nothing is spent, and the numbers below "
            "describe\na scripted mock — not any real model. What is real is "
            "the shape: the same\ndiscovery, the same statistics, the same "
            "frontier you get against your own\nendpoint.",
            title="sweepeval demo",
            expand=False,
        )
    )

    result = asyncio.run(_run(scenario, root if keep else None, profile, runs))
    _finish(result, prefer=prefer, seed=0)

    console.print(
        "\n[dim]That was a simulated target. Point the real thing at your own "
        "endpoint:[/dim]\n  sweepeval run https://your-endpoint --key $KEY"
    )


async def _run(scenario: str, root: Path | None, profile: str, runs: int):
    import tempfile

    import httpx

    from sweepeval.cli.mock import find_scenario
    from sweepeval.execute.sweep import asweep_target
    from sweepeval.mock.app import MockApp
    from sweepeval.mock.scenario import load_scenario

    path = find_scenario(scenario)
    if path is None:
        raise typer.BadParameter(f"no bundled scenario {scenario!r}")

    mock = MockApp(load_scenario(path))
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=mock),
        base_url="https://demo.sweepeval.invalid",
    )

    directory = root
    temp: tempfile.TemporaryDirectory[str] | None = None
    if directory is None:
        temp = tempfile.TemporaryDirectory(prefix="sweepeval-demo-")
        directory = Path(temp.name)

    try:
        return await asweep_target(
            DEMO_URL,
            client=client,
            root=str(directory),
            profile=profile,  # type: ignore[arg-type]
            runs=runs,
            seed=0,
            # The demo's "target" is in-process, so there is nobody to be
            # authorised by and nothing to spend. Both gates are satisfied
            # here rather than skipped in the engine, which would leave a code
            # path where a real run could bypass them.
            authorized=True,
            authorization_prompt=False,
            confirm=None,
        )
    finally:
        await client.aclose()
        if temp is not None and root is None:
            # Held open until the report has been rendered by the caller, so
            # cleanup is deferred rather than dropped.
            _DEFERRED.append(temp)


_DEFERRED: list[object] = []
"""Temp directories kept alive until the process exits.

The report reads artifact paths after the sweep returns, so cleaning up here
would delete the run out from under it.
"""
