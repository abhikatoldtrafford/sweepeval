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

_BUNDLED = Path(__file__).resolve().parents[3] / "tests" / "scenarios"


@app.command()
def serve(
    scenario: str = typer.Option(..., "--scenario", "-s", help="Scenario name or path."),
    port: int = typer.Option(8080, "--port", "-p"),
    host: str = typer.Option("127.0.0.1", "--host"),
) -> None:
    """Serve one scenario on a real port."""
    path = Path(scenario)
    if not path.exists():
        path = _BUNDLED / f"{scenario}.yaml"
    if not path.exists():
        raise typer.BadParameter(f"no scenario {scenario!r} (looked in {_BUNDLED})")

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
