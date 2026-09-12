"""Shape protocol and registry (spec §8.1, I10)."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "SHAPE_ENTRY_POINT_GROUP",
    "Shape",
    "builtin_shapes",
    "ladder_order",
    "register",
    "registry",
]

SHAPE_ENTRY_POINT_GROUP = "sweepeval.discovery_shapes"


@runtime_checkable
class Shape(Protocol):
    """One candidate request shape."""

    name: str
    priority: float
    """Ladder position. Lower is tried first (§8.1's prior order)."""

    default_paths: tuple[str, ...]
    """Suffixes tried during stage C path completion (§8.1)."""

    def build(self, prompt: str, **params: Any) -> dict[str, Any]:
        """Build a request body carrying ``prompt``."""
        ...

    def build_multi_turn(self, turns: list[tuple[str, str]], **params: Any) -> dict[str, Any]:
        """Build a body carrying a (role, text) history."""
        ...

    def prior_text_paths(self) -> tuple[str, ...]:
        """Known response paths for this family, tried before the blind walk."""
        ...


_REGISTRY: dict[str, Shape] = {}


def register(shape: Shape) -> Shape:
    if shape.name in _REGISTRY:
        raise ValueError(f"shape {shape.name!r} is already registered")
    _REGISTRY[shape.name] = shape
    return shape


def registry() -> dict[str, Shape]:
    return dict(_REGISTRY)


def builtin_shapes() -> tuple[Shape, ...]:
    return ladder_order()


def ladder_order(include_entry_points: bool = True) -> tuple[Shape, ...]:
    """Shapes in ladder order: by priority, then name for determinism."""
    shapes = dict(_REGISTRY)
    if include_entry_points:
        for entry in entry_points(group=SHAPE_ENTRY_POINT_GROUP):
            loaded = entry.load()
            shape = loaded() if callable(loaded) and not hasattr(loaded, "name") else loaded
            shapes.setdefault(shape.name, shape)
    return tuple(sorted(shapes.values(), key=lambda s: (s.priority, s.name)))
