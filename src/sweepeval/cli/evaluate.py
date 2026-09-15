"""`sweepeval evaluate`, `baseline` and `gate` (spec §4.1, §16)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from sweepeval.execute.evaluate import (
    EvaluationResult,
    ModelPinRefused,
    aevaluate_target,
)
from sweepeval.execute.gate import (
    DEFAULT_GATE_ON,
    gate_payload,
    load_baseline,
    model_to_pin,
    profile_refusal,
    save_baseline,
    snapshot,
)
from sweepeval.execute.gate import (
    gate as gate_result,
)
from sweepeval.report.machine import (
    as_github_annotations,
    as_json,
    as_markdown,
    gate_annotations,
    in_github_actions,
)
from sweepeval.report.terminal import render_evaluation
from sweepeval.stats.diff import ExitCode

console = Console()

_MODEL_HELP = (
    "Pin the model every probe names. Without it, discovery picks one and a "
    "change to the provider's model list can move a gate onto a different "
    "model between runs."
)


def _run(
    url: str, key: str | None, profile: str, runs: int, root: Path,
    seed: int, authorized: bool, yes: bool = False, model: str | None = None,
) -> EvaluationResult:
    """Run one configuration, behind the same pre-flight a sweep uses (I9)."""
    from sweepeval.cli.sweep import _confirmer

    try:
        result = asyncio.run(
            aevaluate_target(
                url, key=key, profile=profile, runs=runs,  # type: ignore[arg-type]
                root=str(root), seed=seed, authorized=authorized,
                confirm=_confirmer(yes),  # type: ignore[arg-type]
                model=model,
            )
        )
    except ModelPinRefused as error:
        # A usage error, not a comparability refusal: the flag is wrong for
        # this target, which is a fact about the command line.
        console.print(f"[red]REFUSED[/red]    {error}")
        raise typer.Exit(code=int(ExitCode.USAGE_ERROR)) from None
    if result.declined:
        console.print(f"[yellow]{result.declined}[/yellow]")
        raise typer.Exit(code=0)

    # I7: a run that cannot be re-reported from its own directory is not
    # stored, it is merely logged. `evaluate` wrote no aggregates.json at all,
    # so `sweepeval report` on it raised "a run can only be re-reported if it
    # was stored" -- for the single-config case, which is the modal outcome
    # for a target with no sweepable axis.
    from sweepeval.report.stored import write_evaluation

    write_evaluation(result)
    return result


def _emit(result: EvaluationResult, formats: str | None, root: Path) -> None:
    """Machine formats only when asked (§30)."""
    if not formats and not in_github_actions():
        return

    wanted = {f.strip() for f in (formats or "").split(",") if f.strip()}
    if in_github_actions():
        wanted.add("gha")

    store = result.store
    assert store is not None

    for fmt in sorted(wanted):
        if fmt == "json":
            store.report_path("json").write_text(as_json(result), encoding="utf-8")
        elif fmt == "md":
            store.report_path("md").write_text(as_markdown(result), encoding="utf-8")
        elif fmt == "gha":
            text = as_github_annotations(result)
            if text:
                typer.echo(text, nl=False)
        else:
            console.print(f"[yellow]unknown --format {fmt!r}, skipped[/yellow]")

    written = sorted(f for f in wanted if f in {"json", "md"})
    if written:
        console.print(f"\nwrote {', '.join('report.' + f for f in written)} to {store.run_dir}")


def evaluate_command(
    url: str = typer.Argument(..., help="The endpoint to evaluate."),
    key: str | None = typer.Option(None, "--key", "-k"),
    profile: str = typer.Option("quick", "--profile", help="quick | standard | deep"),
    runs: int = typer.Option(3, "-n", "--runs"),
    root: Path = typer.Option(Path(".sweepeval"), "--root"),
    seed: int = typer.Option(0, "--seed"),
    authorized: bool = typer.Option(
        False, "--i-am-authorized",
        help="Affirm you are authorised to run the security suite against this host.",
    ),
    fmt: str | None = typer.Option(None, "--format", help="json,md,gha"),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Accept the pre-flight estimate without prompting."
    ),
    model: str | None = typer.Option(
        None, "--model", help=_MODEL_HELP
    ),
) -> None:
    """Score a single configuration."""
    result = _run(url, key, profile, runs, root, seed, authorized, yes, model)
    render_evaluation(result, console)
    _emit(result, fmt, root)


def baseline_command(
    url: str = typer.Argument(..., help="The endpoint to snapshot."),
    key: str | None = typer.Option(None, "--key", "-k"),
    out: Path = typer.Option(Path("eval/baseline.json"), "--out", "-o"),
    profile: str = typer.Option("standard", "--profile"),
    runs: int = typer.Option(3, "-n", "--runs"),
    root: Path = typer.Option(Path(".sweepeval"), "--root"),
    seed: int = typer.Option(0, "--seed"),
    authorized: bool = typer.Option(False, "--i-am-authorized"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    model: str | None = typer.Option(None, "--model", help=_MODEL_HELP),
) -> None:
    """Snapshot a run as a committable baseline."""
    result = _run(url, key, profile, runs, root, seed, authorized, yes, model)
    render_evaluation(result, console)

    path = save_baseline(snapshot(result), out)
    console.print(f"\nwrote baseline to {path}")
    console.print(
        "[yellow]commit this file.[/yellow] It carries no credentials — the "
        "redactor runs over everything written to it — and the gate needs its "
        "per-cluster values to run a paired test."
    )


def gate_command(
    url: str = typer.Argument(..., help="The endpoint to re-run and compare."),
    baseline_path: Path = typer.Option(..., "--baseline", "-b"),
    key: str | None = typer.Option(None, "--key", "-k"),
    profile: str = typer.Option("standard", "--profile"),
    runs: int = typer.Option(3, "-n", "--runs"),
    root: Path = typer.Option(Path(".sweepeval"), "--root"),
    seed: int = typer.Option(0, "--seed"),
    authorized: bool = typer.Option(False, "--i-am-authorized"),
    gate_on: str | None = typer.Option(
        None, "--gate-on", help=f"Comma-separated. Default: {','.join(DEFAULT_GATE_ON)}"
    ),
    min_effect: list[str] = typer.Option(
        [], "--min-effect", help="metric=value, repeatable."
    ),
    fmt: str | None = typer.Option(None, "--format", help="json,gha"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    model: str | None = typer.Option(
        None, "--model",
        help="Pin the model. Defaults to the baseline's, so the gate "
             "re-measures what the baseline measured.",
    ),
    allow_model_change: bool = typer.Option(
        False, "--allow-model-change",
        help="Compare against a baseline taken on a different model. The "
             "verdict says so; a difference may be the model, not the change.",
    ),
) -> None:
    """Re-run the target and compare against a baseline. Exits 0/1/2/3."""
    # Before `_run`, which spends. Refusing an ineligible profile after paying
    # for the evaluation it refuses to use cost 120 live requests the first
    # time this path was exercised against a real endpoint.
    refusal = profile_refusal(profile)
    if refusal is not None:
        # ExitCode.USAGE_ERROR, the same code `gate` returns for this refusal
        # after the run. Moving the check earlier must not change what CI sees.
        console.print(f"[red]REFUSED[/red]    {refusal}")
        raise typer.Exit(code=int(ExitCode.USAGE_ERROR))

    baseline = load_baseline(baseline_path)

    pinned = model_to_pin(baseline, model, allow_model_change)
    if pinned is not None and model is None:
        console.print(f"[dim]pinned to the baseline's model: {pinned}[/dim]")
    model = pinned

    result = _run(url, key, profile, runs, root, seed, authorized, yes, model)

    overrides: dict[str, float] = {}
    for entry in min_effect:
        name, _, value = entry.partition("=")
        try:
            overrides[name.strip()] = float(value)
        except ValueError:
            raise typer.BadParameter(f"--min-effect wants metric=value, got {entry!r}") from None

    try:
        verdict = gate_result(
            result, baseline,
            gate_on=tuple(g.strip() for g in gate_on.split(",")) if gate_on else None,
            min_effect_overrides=overrides or None,
            seed=seed,
            allow_model_change=allow_model_change,
        )
    except ValueError as error:
        # An unknown --gate-on name. The engine raises so a typo cannot gate
        # nothing silently, and the CLI has to turn that into a usage error
        # the user can read -- it escaped as a traceback, which buries the
        # message naming the metric and the alternatives.
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(code=ExitCode.USAGE_ERROR.value) from None

    console.print(verdict.explain())

    wanted = {f.strip() for f in (fmt or "").split(",") if f.strip()}
    if in_github_actions():
        wanted.add("gha")
    payload = gate_payload(verdict)
    if "gha" in wanted:
        text = gate_annotations(payload)
        if text:
            typer.echo(text, nl=False)
    if "json" in wanted and result.store is not None:
        import json as _json

        # `run_dir / "gate.json"`, not `report_path("gate.json")` -- that
        # helper prefixes "report.", so the payload landed as
        # `report.gate.json` while the README told CI to read `gate.json`.
        # A job following the docs found nothing, and could not tell that
        # from a gate that had written nothing.
        (result.store.run_dir / "gate.json").write_text(
            _json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    raise typer.Exit(code=verdict.exit_code.value)


def init_command(
    root: Path = typer.Option(Path("."), "--root"),
) -> None:
    """Write .gitignore entries and say what is safe to commit (§6.6)."""
    gitignore = Path(root) / ".gitignore"
    entry = ".sweepeval/"
    existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""

    if entry not in existing:
        with gitignore.open("a", encoding="utf-8") as handle:
            if existing and not existing.endswith("\n"):
                handle.write("\n")
            handle.write(f"\n# sweepeval run artifacts (large, regenerable)\n{entry}\n")
        console.print(f"added {entry} to {gitignore}")
    else:
        console.print(f"{entry} already in {gitignore}")

    console.print(
        "\n[bold]safe to commit[/bold]\n"
        "  eval/baseline.json   the gate needs it, and it carries no credentials\n"
        "  sweepeval.yaml       the annotated config; correct it by hand\n"
        "\n[bold]not committed[/bold]\n"
        "  .sweepeval/          run artifacts: calls, observations, blobs\n"
    )


def _exit_code_help() -> str:
    return " ".join(f"{c.value}={c.name.lower()}" for c in ExitCode)
