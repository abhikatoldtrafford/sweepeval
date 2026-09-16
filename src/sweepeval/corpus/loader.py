"""Corpus loading, profile selection and the cluster-floor check (spec §10.2, §10.3)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sweepeval.corpus.template import PROFILES, ProbeTemplate, Profile
from sweepeval.schema.hashing import corpus_hash

__all__ = [
    "CLUSTER_KEY_BY_FAMILY",
    "SUITE_ROOT",
    "Corpus",
    "load_corpus",
]

SUITE_ROOT = Path(__file__).parent / "suites"

# Which clusters each family's objective resamples over (§13.3).
CLUSTER_KEY_BY_FAMILY: dict[str, str] = {
    "security": "security_probe",
    "guardrail": "guardrail_probe",
    "determinism": "determinism_base_prompt",
    "context": "conversation",
    "degradation": "degradation_probe",
    "tool_integrity": "tool_probe",
}


@dataclass
class Corpus:
    """A loaded, profile-filtered probe corpus."""

    templates: tuple[ProbeTemplate, ...]
    suite: str
    suite_version: int
    profile: Profile
    hash: str

    def by_family(self, family: str) -> tuple[ProbeTemplate, ...]:
        return tuple(t for t in self.templates if t.family == family)

    @property
    def probes(self) -> tuple[ProbeTemplate, ...]:
        """Scored probes only.

        Excludes ``family: operational`` instrument prompts — the two
        open-ended prompts the sampling-effect test uses (§9.1). They live in
        the corpus so a change to them changes the corpus hash, but they are
        not scored, are not units, and are charged to the capability budget
        rather than the scoring estimate.
        """
        return tuple(t for t in self.templates if t.family != "operational")

    @property
    def ambiguity_capable(self) -> tuple[ProbeTemplate, ...]:
        """Probes that can reach the judge (§11.9).

        A contract with no ``ambiguous_when`` never returns AMBIGUOUS, so it
        never escalates -- which is what makes the worst-case judge estimate
        bounded and honest rather than "every probe, maybe".
        """
        return tuple(
            t
            for t in self.probes
            if any(s.ambiguous_when for s in t.scoring)
        )

    @property
    def instruments(self) -> tuple[ProbeTemplate, ...]:
        return tuple(t for t in self.templates if t.family == "operational")

    @property
    def unit_count(self) -> int:
        return len(self.probes)

    @property
    def calls_per_run(self) -> int:
        return sum(t.calls_per_run for t in self.probes)

    def cluster_count(self, family: str) -> int:
        """Clusters this family's objective resamples over (§13.3).

        Not the same as the unit count. Determinism's clusters are the base
        prompts alone — the invariance groups belong to a different
        sub-scorer — and that distinction is exactly why the family carries 12
        base prompts rather than 8, which would sit on the floor.
        """
        templates = self.by_family(family)
        if family == "determinism":
            return sum(1 for t in templates if t.group is None)
        if family == "degradation":
            # Two metrics, each resampling over its own probes, so the floor
            # binds on the smaller of the two rather than on their sum. Twenty
            # probes that were seventeen long-input and three load would clear
            # a family-level check while the load interval was meaningless.
            kinds = [t.degradation_kind for t in templates]
            return min(kinds.count("long_input"), kinds.count("load"))
        return len(templates)

    def depth_strata(self) -> dict[int, int]:
        """Conversations per depth. The AUC bootstrap stratifies on these."""
        counts: dict[int, int] = {}
        for template in self.by_family("context"):
            if template.depth is not None:
                counts[template.depth] = counts.get(template.depth, 0) + 1
        return dict(sorted(counts.items()))

    def estimate(self, configs: int, runs: int) -> dict[str, int]:
        """§12.3's scoring line: sum of calls_per_run, not units x runs.

        ``configs x units x runs`` is wrong for every multi-turn unit, and it
        is the number the user consents to under I9.
        """
        return {
            "units": self.unit_count,
            "calls_per_run": self.calls_per_run,
            "total_calls": self.calls_per_run * runs * configs,
        }


def load_corpus(
    profile: Profile = "standard",
    root: Path | None = None,
    suite: str = "generic",
    version: int = 1,
) -> Corpus:
    """Load a suite and filter to one profile."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")

    directory = (root or SUITE_ROOT) / suite / f"v{version}"
    if not directory.is_dir():
        raise FileNotFoundError(f"no suite at {directory}")

    all_templates: list[ProbeTemplate] = []
    raw_bytes: list[bytes] = []

    for path in sorted(directory.glob("*.yaml")):
        raw = path.read_bytes()
        payload = yaml.safe_load(raw.decode("utf-8"))
        if payload is None:
            continue
        if not isinstance(payload, list):
            raise ValueError(f"{path}: a corpus file must be a list of templates")
        raw_bytes.append(raw)
        for entry in payload:
            all_templates.append(ProbeTemplate.model_validate(entry))

    _check_unique_ids(all_templates)

    selected = tuple(t for t in all_templates if profile in t.profiles)

    return Corpus(
        templates=selected,
        suite=suite,
        suite_version=version,
        profile=profile,
        # The hash covers the WHOLE suite plus the profile definitions, not the
        # filtered subset: two profiles of the same corpus must agree that the
        # corpus is the same one (§10.3).
        hash=corpus_hash(raw_bytes, version, _profile_definitions(all_templates)),
    )


def _check_unique_ids(templates: Iterable[ProbeTemplate]) -> None:
    seen: set[str] = set()
    for template in templates:
        if template.id in seen:
            raise ValueError(f"duplicate template id {template.id!r}")
        seen.add(template.id)


def _profile_definitions(templates: Iterable[ProbeTemplate]) -> dict[str, Any]:
    definitions: dict[str, list[str]] = {p: [] for p in PROFILES}
    for template in templates:
        for profile in template.profiles:
            definitions[profile].append(template.id)
    return {k: sorted(v) for k, v in definitions.items()}
