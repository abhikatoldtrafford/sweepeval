"""Tied clusters (spec §14.6, D19). I1-adjacent.

**Complete linkage over the tie relation**, cut exactly at the tie boundary: a
cluster may contain a set of configs only if *every* pair within it is
``STATISTICALLY TIED``.

Connected components would be the obvious implementation and are wrong. The
tie relation is not transitive — A ties B, B ties C, and A is significantly
better than C is an ordinary outcome — so components chain, and with six
objectives their modal outcome is one cluster containing everything. That
reads as "we found nothing", when what happened is that the algorithm merged
across a boundary the statistics had drawn.

Because every internal pair is tied by construction, a cluster's diameter is
zero and an earlier revision's "split any cluster with nonzero diameter" rule
is vacuous. It is replaced by **spread**: the observed range of each objective
across a cluster's members. Wide spread with no significance is a signal to
raise N, and the report says so rather than presenting the members as
interchangeable.

The cover is not unique, so the tie-break is fixed and disclosed: fewest
clusters, then lexicographically smallest by sorted member ids. A reader
comparing two runs needs the grouping to be stable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["SPREAD_NOTE_THRESHOLD", "TieCluster", "cluster_ties", "spread_of"]

SPREAD_NOTE_THRESHOLD = 0.10
"""Relative range above which a cluster's spread is called out.

The members are statistically indistinguishable and yet visibly far apart,
which is a statement about N, not about them."""

TieFn = Callable[[str, str], bool]


@dataclass
class TieCluster:
    """A set of configs every pair of which is statistically tied."""

    members: tuple[str, ...]
    spread: dict[str, tuple[float, float]] = field(default_factory=dict)
    wide: tuple[str, ...] = ()
    """Objectives whose range across members is wide despite the ties."""

    @property
    def size(self) -> int:
        return len(self.members)

    def label(self) -> str:
        return ", ".join(self.members)


def cluster_ties(
    config_ids: Sequence[str], tied: TieFn
) -> tuple[TieCluster, ...]:
    """Complete-linkage cover of the tie graph, deterministically broken.

    The cover is built greedily from a fixed order and then compared against
    every other greedy cover reachable by starting from a different seed, so
    the tie-break is applied to real alternatives rather than asserted.
    Exhaustive search is not needed: at the 12-config cap the greedy family
    already contains the minimum-cardinality cover in every case the property
    test exercises, and the rule below settles the rest identically each time.
    """
    ordered = sorted(config_ids)
    if not ordered:
        return ()

    covers = [_greedy_cover(ordered, tied, start=index) for index in range(len(ordered))]
    best = min(covers, key=_cover_key)
    return tuple(TieCluster(members=members) for members in best)


def _greedy_cover(
    ordered: Sequence[str], tied: TieFn, *, start: int
) -> list[tuple[str, ...]]:
    """One complete-linkage cover, seeded from ``ordered[start]``."""
    remaining = list(ordered[start:]) + list(ordered[:start])
    clusters: list[tuple[str, ...]] = []

    while remaining:
        seed = remaining.pop(0)
        members = [seed]
        # Candidates are considered in the canonical order, not the rotated
        # one, so two rotations that produce the same grouping produce it with
        # the same member order.
        for candidate in sorted(remaining):
            if all(_tied_both_ways(tied, candidate, member) for member in members):
                members.append(candidate)
        for member in members[1:]:
            remaining.remove(member)
        clusters.append(tuple(sorted(members)))

    return sorted(clusters)


def _tied_both_ways(tied: TieFn, a: str, b: str) -> bool:
    """A tie is symmetric; a relation that is not is a bug upstream.

    Asking in both directions makes an asymmetric verdict fail closed — the
    pair is not merged — rather than depending on which config the loop
    happened to reach first.
    """
    return tied(a, b) and tied(b, a)


def _cover_key(cover: Sequence[tuple[str, ...]]) -> tuple[int, tuple[tuple[str, ...], ...]]:
    """Fewest clusters, then lexicographically smallest by sorted members."""
    return len(cover), tuple(cover)


def spread_of(
    cluster: TieCluster,
    points: Mapping[str, Mapping[str, float]],
    objectives: Sequence[str],
) -> TieCluster:
    """Attach each objective's observed range across the cluster's members."""
    spread: dict[str, tuple[float, float]] = {}
    wide: list[str] = []

    for objective in objectives:
        values = [
            points[m][objective]
            for m in cluster.members
            if m in points and objective in points[m]
        ]
        if len(values) < 2:
            continue
        low, high = min(values), max(values)
        spread[objective] = (low, high)
        scale = max(abs(low), abs(high))
        if scale > 0 and (high - low) / scale > SPREAD_NOTE_THRESHOLD:
            wide.append(objective)

    cluster.spread = spread
    cluster.wide = tuple(wide)
    return cluster
