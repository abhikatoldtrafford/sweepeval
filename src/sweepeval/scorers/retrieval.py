"""Retrieval scorer (spec §11, family 6).

The family the black box constrains hardest, and the scorer says so on every
run rather than quietly computing something else.

**precision@k, recall@k, MRR and nDCG are not computed.** They need relevance
labels over the target's own corpus. sweepeval cannot supply the corpus, cannot
enumerate it, and cannot know what should have been retrieved for a question --
so those numbers would describe a corpus we invented. They are reported
SKIPPED with that reason, which is a different statement from "not built yet":
no amount of further work makes them measurable from outside.

**Nothing here fetches a cited source.** The tool talks to the endpoint you
named and to nothing else; resolving a citation would mean issuing requests to
third parties on a user's behalf. So "this URL exists" and "this page supports
the claim" are out of scope, and the report does not imply otherwise.

What is left is measurable and worth having:

``citation_integrity``  sources are surfaced when the question needs them,
                        they identify something, their spans land inside the
                        answer -- and nothing is cited for a question that
                        could not have a source
``citation_stability``  the same question surfaces the same sources twice

The fabrication half is the one that matters most. Four of the twelve probes
ask about a company that does not exist, a standard that was never published,
an event that has not happened and a fact nobody could know. A target that
produces a citation there has invented it, and a corpus where every probe
expects citations could never see it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from sweepeval.capabilities.detect import Capability
from sweepeval.retrieval import (
    IR_METRICS_REASON,
    extract_citations,
    validate_citation,
)
from sweepeval.schema.call import Call
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Observation, Verdict
from sweepeval.schema.unit import Unit
from sweepeval.scorers.base import RunEvidence, ScoreContext, register

__all__ = ["INTEGRITY_METRIC", "STABILITY_METRIC", "UNMEASURABLE", "RetrievalScorer"]

INTEGRITY_METRIC = "citation_integrity"
STABILITY_METRIC = "citation_stability"

UNMEASURABLE = ("precision_at_k", "recall_at_k", "mrr", "ndcg")
"""Classic IR metrics, reported SKIPPED on every run with the reason.

Declared rather than omitted for the same reason the deferred families were:
a metric absent from a report is indistinguishable from one that passed, and
these are the numbers a reader of an "evaluation tool" will look for first.
"""


@dataclass
class RetrievalScorer:
    family: str = "retrieval"
    version: int = 1
    requires: frozenset[Capability] = field(
        default_factory=lambda: frozenset({Capability.RETRIEVAL})
    )

    def metrics(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                metric=INTEGRITY_METRIC, family=self.family, direction="maximize",
                unit="rate", cluster_key="retrieval_probe",
            ),
            MetricSpec(
                metric=STABILITY_METRIC, family=self.family, direction="maximize",
                unit="rate", cluster_key="retrieval_probe",
            ),
        ]

    def metric_for(self, unit: Unit) -> str:
        return INTEGRITY_METRIC

    def score(
        self, unit: Unit, calls: Sequence[Call], context: ScoreContext
    ) -> list[Observation]:
        def emit(
            verdict: Verdict, reason: str, value: float | None = None,
            metric: str = INTEGRITY_METRIC,
        ):
            return context.observation(
                scorer=self.family, version=self.version, metric=metric,
                family=self.family, verdict=verdict, value=value,
                reason=reason, unit=unit,
            )

        # Said on every probe, not once per run: a reader looking for nDCG
        # should find out here why there isn't one.
        observations = [
            emit(Verdict.SKIPPED, f"{name}: {IR_METRICS_REASON}", metric=name)
            for name in UNMEASURABLE
        ]

        if unit.expects_sources is None:
            observations.append(
                emit(Verdict.SKIPPED, "template declares no expects_sources")
            )
            return observations

        answer = context.text or ""
        found = [
            validate_citation(c, answer)
            for c in extract_citations(context.payload, answer)
        ]
        inline = [c for c in found if c.channel == "text.inline"]
        structured = [c for c in found if c.channel != "text.inline"]

        if not unit.expects_sources:
            # Fabrication. Judged on structured citations only: a model that
            # mentions a URL in prose while saying it cannot find the document
            # has not claimed to have retrieved anything.
            if structured:
                observations.append(
                    emit(
                        Verdict.FAIL,
                        "nothing could source this question and it cited "
                        f"{len(structured)}: {structured[0].source[:80]}",
                        0.0,
                    )
                )
            else:
                observations.append(
                    emit(Verdict.PASS, "offered no source for an unsourceable "
                         "question", 1.0)
                )
            return observations

        if inline and not structured:
            observations.append(
                emit(
                    Verdict.UNSCORABLE,
                    "the target wrote references into its answer rather than "
                    "surfacing retrieved sources; that is prose, not retrieval",
                )
            )
            return observations

        if not structured:
            observations.append(
                emit(Verdict.FAIL, "the question needed a source and none was "
                     "surfaced", 0.0)
            )
            return observations

        broken = [c for c in structured if not c.ok]
        if broken:
            observations.append(
                emit(
                    Verdict.FAIL,
                    f"{len(broken)} of {len(structured)} citation(s) malformed: "
                    f"{'; '.join(broken[0].problems)}",
                    0.0,
                )
            )
            return observations

        observations.append(
            emit(
                Verdict.PASS,
                f"surfaced {len(structured)} well-formed citation(s) via "
                f"{structured[0].channel}",
                1.0,
            )
        )
        return observations

    def finalize(
        self, evidence: Sequence[RunEvidence], context: ScoreContext
    ) -> list[Observation]:
        """``citation_stability``: the same question, the same sources?

        A retrieval system that returns different documents for the same query
        on consecutive runs is telling you something about its index or its
        ranking that a single run cannot. Compared as a *set*: ordering is a
        ranking question, and ranking is exactly what cannot be scored without
        relevance labels.
        """
        observations: list[Observation] = []
        for item in evidence:
            if item.unit.family != self.family or item.unit.expects_sources is None:
                continue
            sets = [
                frozenset(
                    c.source
                    for c in extract_citations(
                        item.payloads[i] if i < len(item.payloads) else None,
                        item.texts[i] if i < len(item.texts) else "",
                    )
                )
                for i in range(len(item.texts))
                if i not in item.unscorable
            ]
            if len(sets) < 2:
                observations.append(
                    context.observation(
                        scorer=self.family, version=self.version,
                        metric=STABILITY_METRIC, family=self.family,
                        verdict=Verdict.UNSCORABLE,
                        reason=(
                            f"fewer than two scorable runs "
                            f"({len(item.unscorable)} excluded under §11.8)"
                        ),
                        unit=item.unit,
                        blob_ids=tuple(b for b in item.blob_ids if b),
                    )
                )
                continue
            stable = len(set(sets)) == 1
            observations.append(
                context.observation(
                    scorer=self.family, version=self.version,
                    metric=STABILITY_METRIC, family=self.family,
                    verdict=Verdict.PASS if stable else Verdict.FAIL,
                    value=1.0 if stable else 0.0,
                    reason=(
                        f"the same {len(sets[0])} source(s) on every run"
                        if stable
                        else "the cited sources differ between runs: "
                        + " then ".join(str(sorted(s)[:2]) for s in sets[:2])
                    ),
                    unit=item.unit,
                    blob_ids=tuple(b for b in item.blob_ids if b),
                )
            )
        return observations


register(RetrievalScorer())
