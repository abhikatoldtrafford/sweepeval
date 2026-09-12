"""Request shapes (spec §8.1, I10).

A shape knows how to build a request body and how to name itself. The ladder
tries them in prior order; out-of-tree shapes declare a ``priority`` and are
inserted by it, so a third-party shape has a defined position rather than
landing wherever registration happened to run.

New endpoint shapes are the likeliest external contribution (§19.2), so this is
the plugin surface documented first and best.
"""

from __future__ import annotations

from sweepeval.discovery.shapes.base import (
    SHAPE_ENTRY_POINT_GROUP,
    Shape,
    builtin_shapes,
    ladder_order,
    register,
    registry,
)

__all__ = [
    "SHAPE_ENTRY_POINT_GROUP",
    "Shape",
    "builtin_shapes",
    "ladder_order",
    "register",
    "registry",
]
