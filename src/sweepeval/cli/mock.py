"""`sweepeval mock serve` (spec §17).

The same ASGI app the test suite mounts in-process, exposed on a real port for
manual work and for exercising the socket path itself.
"""

from __future__ import annotations

from pathlib import Path

import typer

from sweepeval.mock.app import MockApp
from sweepeval.mock.scenario import load_scenario

app = typer.Typer(help="Run the scenario mock endpoint.", no_args_is_help=True)

_PACKAGED = Path(__file__).resolve().parent.parent / "mock" / "scenarios"
_REPO = Path(__file__).resolve().parents[3] / "tests" / "scenarios"


def find_scenario(name: str) -> Path | None:
    """Packaged scenarios first, then the repo's test scenarios.

    An installed user has no ``tests/`` directory, so a bundled scenario has
    to live inside the package or ``mock serve`` and ``demo`` work only from a
    git checkout — which is exactly the kind of thing that passes in CI and
    fails for every user.
    """
    direct = Path(name)
    if direct.exists():
        return direct
    for root in (_PACKAGED, _REPO):
        candidate = root / f"{name}.yaml"
        if candidate.exists():
            return candidate
    return None


@app.command()
def serve(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name or path."),
    port: int = typer.Option(8080, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Serve one scenario on a real port."""
    path = find_scenario(scenario)
    if path is None:
        raise typer.BadParameter(
            f"no scenario {scenario!r} (looked in {_PACKAGED} and {_REPO})"
        )

    loaded = load_scenario(path)
    mock = MockApp(loaded)

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise typer.BadParameter(
            "mock serve needs uvicorn: pip install 'sweepeval[dev]' or pip install uvicorn"
        ) from exc

    typer.echo(f"serving scenario {loaded.name!r} on http://{host}:{port}")
    for probe_path in loaded.paths:
        typer.echo(f"  POST {probe_path}")
    uvicorn.run(mock, host=host, port=port, log_level="warning")
