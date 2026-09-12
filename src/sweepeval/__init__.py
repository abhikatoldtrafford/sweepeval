"""sweepeval — zero-config, black-box sweep and benchmark engine."""

__version__ = "0.1.0.dev0"

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

__all__ = [
    "__version__", "baseline", "compare", "demo", "discover", "evaluate",
    "gate", "rank", "report", "run", "run_gate", "sweep",
]
