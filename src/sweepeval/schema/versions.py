"""Version constants (spec §6.5).

``SCHEMA_MAJOR`` is a hard comparability key: results written under different
major versions refuse to be compared. ``SUITE_VERSION`` and ``TOOL_VERSION``
are recorded in every manifest for reproducibility (§13.8).

``TOOL_VERSION`` is read from the installed distribution metadata rather than
written down here. There were three hardcoded copies of it -- this module,
``sweepeval.__init__`` and ``pyproject.toml`` -- and nothing kept them in
step. A stale one is not cosmetic: the version goes into every manifest as a
soft comparability key, so two runs of genuinely different code could claim
the same one, and ``sweepeval version`` could disagree with what pip
installed. ``pyproject.toml`` is now the single source.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from typing import Final

SCHEMA_MAJOR: Final = 1
SCHEMA_VERSION: Final = "1.0"
SUITE_VERSION: Final = 1


def _tool_version() -> str:
    try:
        return _distribution_version("sweepeval")
    except PackageNotFoundError:  # pragma: no cover - only when run uninstalled
        # Running from a source tree with nothing installed. Say so rather
        # than guessing a number, because the guess would be recorded in a
        # manifest as though it were the version that produced the results.
        return "0+unknown"


TOOL_VERSION: Final = _tool_version()
