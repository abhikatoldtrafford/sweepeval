"""Report renderers (spec §15). Tier 2 plugin surface.

Renderers land in M9; the protocol is declared here so the API-surface
snapshot pins it from M0.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["Reporter"]


@runtime_checkable
class Reporter(Protocol):
    """Renders a frontier into one output format."""

    fmt: str

    def render(self, frontier: Any, aggregates: Any, manifest: Any) -> str:
        """Return the rendered report."""
        ...
