"""CLI entry point (spec §4.1).

The CLI is a thin caller of :mod:`sweepeval.api`; no logic lives only here.
Verbs are registered by the milestone that implements them, so this module
stays a router.
"""

from __future__ import annotations

import typer

from sweepeval import __version__

app = typer.Typer(
    name="sweepeval",
    help="Zero-config, black-box sweep and benchmark engine for LLM and agent systems.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def _main() -> None:
    """Keep the app a command group.

    Without an explicit callback, Typer collapses a single registered command
    into the root, so ``sweepeval version`` fails to route. Verbs are added by
    later milestones; this keeps routing correct from the first one.
    """


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
