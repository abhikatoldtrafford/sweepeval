"""Determinism scorers (spec §11.4).

Three scorers, never collapsed into one number.

The naming matters more here than anywhere else in the tool.
``target_determinism_at_temp0`` is a property of the (model, system_prompt)
pair; ``config_repeatability`` is a property of the row it sits on. An earlier
revision measured one thing and labelled it the other, so a ``temp=1.0`` config
displayed ``0.95`` — a number taken at ``temp=0``, next to a configuration that
is near-0% repeatable in practice.

Where temperature is not a swept axis the two coincide, and §11.4 says so.
Both are still reported, because the distinction has to survive the moment a
temperature axis appears.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import combinations

from sweepeval.capabilities.detect import Capability
from sweepeval.capabilities.normalise import normalise, token_jaccard
from sweepeval.schema.call import Call
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import RunEvidence, ScoreContext, register

__all__ = ["DeterminismScorer"]


@dataclass
class DeterminismScorer:
    family: str = "determinism"
    version: int = 1
    requires: frozenset[Capability] = field(default_factory=frozenset)
    temperature_is_swept: bool = False
    """When False, ``target_determinism_at_temp0`` and ``config_repeatability``
    are the same measurement and §11.4 permits reporting them as such."""

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                metric="target_determinism_at_temp0", family="determinism",
                direction="maximize", unit="rate",
                cluster_key="determinism_base_prompt",
            ),
            MetricSpec(
                metric="config_repeatability", family="determinism",
                direction="maximize", unit="rate",
                cluster_key="determinism_base_prompt",
            ),
            MetricSpec(
                metric="semantic_stability", family="determinism",
                direction="maximize", unit="similarity",
                cluster_key="determinism_base_prompt",
            ),
            MetricSpec(
                metric="invariance", family="determinism",
                direction="maximize", unit="similarity",
                cluster_key="determinism_base_prompt",
            ),
        ]

    def score(
        self, unit: Unit, calls: Sequence[Call], context: ScoreContext
    ) -> list[Observation]:
        """Nothing per-run. Determinism is a statement about the set of runs."""
        return []

    def finalize(
        self, evidence: Sequence[RunEvidence], context: ScoreContext
    ) -> list[Observation]:
        observations: list[Observation] = []
        by_group: dict[str, list[RunEvidence]] = {}

        for item in evidence:
            if item.unit.family != "determinism":
                continue
            group = item.unit.params.get("group") or _group_of(item.unit)
            if group:
                by_group.setdefault(str(group), []).append(item)
                continue
            observations.extend(self._base_prompt(item, context))

        for group, members in sorted(by_group.items()):
            observations.extend(self._invariance(group, members, context))

        return observations

    # --- base prompts: repeatability and semantic stability ---------------

    def _base_prompt(
        self, item: RunEvidence, context: ScoreContext
    ) -> list[Observation]:
        texts = [
            t for i, t in enumerate(item.texts) if i not in item.unscorable and t
        ]
        if len(texts) < 2:
            # §11.8: a refused or failed run leaves the metric, and one
            # surviving run cannot say anything about repeatability.
            #
            # target_determinism_at_temp0 is in this list because it was not,
            # and an excluded trial then produced no row for it at all --
            # missing data where UNSCORABLE was meant, which I5 forbids and
            # which the coverage check cannot see.
            metrics = ["config_repeatability", "semantic_stability"]
            if not self.temperature_is_swept:
                metrics.append("target_determinism_at_temp0")
            excluded = len(item.unscorable)
            return [
                context.observation(
                    scorer=self.family, version=self.version, metric=metric,
                    family=self.family, verdict=Verdict.UNSCORABLE,
                    reason=(
                        f"fewer than two scorable runs "
                        f"({excluded} excluded under §11.8)"
                    ),
                    unit=item.unit,
                    blob_ids=_blobs(item),
                )
                for metric in metrics
            ]

        exact = _exact_match_rate(texts)
        semantic = _mean_pairwise_similarity(texts)

        observations = [
            context.observation(
                scorer=self.family, version=self.version,
                metric="config_repeatability", family=self.family,
                verdict=Verdict.PASS, value=exact,
                reason=f"{len(texts)} scorable runs at the config's own settings",
                unit=item.unit, blob_ids=_blobs(item),
            ),
            context.observation(
                scorer=self.family, version=self.version,
                metric="semantic_stability", family=self.family,
                verdict=Verdict.PASS, value=semantic,
                reason="mean pairwise lexical similarity (D12 default backend)",
                unit=item.unit, blob_ids=_blobs(item),
            ),
        ]

        if not self.temperature_is_swept:
            # §11.4: with no temperature axis the two definitions coincide.
            # Emitted as its own metric rather than aliased, so the moment an
            # axis appears the distinction is already in the data.
            observations.append(
                context.observation(
                    scorer=self.family, version=self.version,
                    metric="target_determinism_at_temp0", family=self.family,
                    verdict=Verdict.PASS, value=exact,
                    reason=(
                        "temperature is not a swept axis, so this coincides "
                        "with config_repeatability (§11.4)"
                    ),
                    unit=item.unit, blob_ids=_blobs(item),
                )
            )

        return observations

    # --- invariance groups -------------------------------------------------

    def _invariance(
        self, group: str, members: Sequence[RunEvidence], context: ScoreContext
    ) -> list[Observation]:
        """Paraphrases of one request must produce equivalent answers."""
        firsts = [
            next((t for i, t in enumerate(m.texts) if i not in m.unscorable and t), "")
            for m in members
        ]
        usable = [t for t in firsts if t]

        if len(usable) < 2:
            return [
                context.observation(
                    scorer=self.family, version=self.version, metric="invariance",
                    family=self.family, verdict=Verdict.UNSCORABLE,
                    reason=f"group {group}: fewer than two scorable paraphrases",
                    unit=members[0].unit,
                    blob_ids=tuple(b for m in members for b in _blobs(m)),
                )
            ]

        return [
            context.observation(
                scorer=self.family, version=self.version, metric="invariance",
                family=self.family, verdict=Verdict.PASS,
                value=_mean_pairwise_similarity(usable),
                reason=f"group {group}: {len(usable)} paraphrases compared",
                unit=members[0].unit,
                # Every paraphrase in the group, since the verdict is about
                # all of them together.
                blob_ids=tuple(b for m in members for b in _blobs(m)),
            )
        ]


def _blobs(item: RunEvidence) -> tuple[str, ...]:
    """Blob addresses of the runs this verdict actually compared.

    Excluded and empty runs are dropped: a blob id for a text that was not
    scored points a reader at evidence the number does not rest on.
    """
    return tuple(
        blob
        for i, blob in enumerate(item.blob_ids)
        if blob and i not in item.unscorable
    )


def _group_of(unit: Unit) -> str | None:
    """Recover the invariance group from the template id.

    ``det.inv.<group>.<variant>.v1`` — the group is not a slot, so it does not
    reach ``Unit.params``.
    """
    parts = unit.template_id.split(".")
    if len(parts) >= 4 and parts[0] == "det" and parts[1] == "inv":
        return parts[2]
    return None


def _exact_match_rate(texts: Sequence[str]) -> float:
    """Fraction of run pairs that are byte-identical after normalisation.

    Normalised, so a target that stamps a timestamp or echoes a request id is
    not counted as non-deterministic for that alone (§9.1's normaliser).
    """
    pairs = list(combinations(texts, 2))
    if not pairs:
        return 1.0
    matches = sum(1 for a, b in pairs if normalise(a) == normalise(b))
    return matches / len(pairs)


def _mean_pairwise_similarity(texts: Sequence[str]) -> float:
    pairs = list(combinations(texts, 2))
    if not pairs:
        return 1.0
    return sum(token_jaccard(a, b) for a, b in pairs) / len(pairs)


register(DeterminismScorer())
