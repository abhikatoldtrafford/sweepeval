"""Version constants (spec §6.5).

``SCHEMA_MAJOR`` is a hard comparability key: results written under different
major versions refuse to be compared. ``SUITE_VERSION`` and ``TOOL_VERSION``
are recorded in every manifest for reproducibility (§13.8).
"""

from __future__ import annotations

from typing import Final

SCHEMA_MAJOR: Final = 1
SCHEMA_VERSION: Final = "1.0"
SUITE_VERSION: Final = 1
TOOL_VERSION: Final = "0.1.0.dev0"
