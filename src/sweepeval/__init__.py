"""sweepeval — zero-config, black-box sweep and benchmark engine."""

from sweepeval.api import (
    baseline,
    compare,
    demo,
    discover,
    evaluate,
    gate,
    rank,
    report,
    run,
    run_gate,
    sweep,
)
from sweepeval.schema.versions import TOOL_VERSION as __version__

__all__ = [
    "__version__", "baseline", "compare", "demo", "discover", "evaluate",
    "gate", "rank", "report", "run", "run_gate", "sweep",
]
