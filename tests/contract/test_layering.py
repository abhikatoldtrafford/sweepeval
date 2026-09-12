"""Layering contracts (spec §5.2).

These are structural invariants, not style preferences. They are the mechanism
behind several of the spec's guarantees: only ``http`` performs requests, only
``store`` writes artifacts, only ``rank`` decides domination, only ``stats``
computes intervals, and ``schema`` is a leaf that imports nothing from the
package.
"""

from __future__ import annotations

import configparser
import pathlib
import re

from importlinter.cli import EXIT_STATUS_SUCCESS, lint_imports

CONFIG = pathlib.Path(".importlinter")


def test_import_linter_contracts_hold() -> None:
    """Run the linter in-process.

    Deliberately NOT ``subprocess.run([sys.executable, "-m", "importlinter.cli",
    ...])``: that module has no ``__main__`` guard, so it imports cleanly, prints
    nothing, and exits 0 no matter how badly the contracts are violated. A check
    that cannot fail is worse than no check.
    """
    assert lint_imports(config_filename=str(CONFIG)) == EXIT_STATUS_SUCCESS


def test_every_declared_contract_is_actually_evaluated() -> None:
    """Guard against a silently-skipped contract.

    ``lint_imports`` returns success both when all contracts pass and when a
    misconfiguration means none were run. Assert the count the linter reports
    matches the count declared in the config file.
    """
    parser = configparser.ConfigParser()
    parser.read(CONFIG, encoding="utf-8")
    declared = [s for s in parser.sections() if s.startswith("importlinter:contract:")]
    assert len(declared) >= 5, f"expected the five layering contracts, found {declared}"

    for section in declared:
        assert parser[section].get("type"), f"{section} declares no contract type"
        assert parser[section].get("source_modules"), f"{section} declares no sources"


def test_schema_imports_nothing_from_package() -> None:
    """schema/ is a leaf.

    import-linter covers this too, but this test states the rule in a form that
    fails with a readable message naming the offending file and module.
    """
    root = pathlib.Path("src/sweepeval/schema")
    assert root.is_dir(), "src/sweepeval/schema must exist"

    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"^\s*(?:from|import)\s+(sweepeval\S*)", text, re.M):
            module = match.group(1)
            if not module.startswith("sweepeval.schema"):
                offenders.append(f"{path}: {module}")

    assert offenders == [], offenders
