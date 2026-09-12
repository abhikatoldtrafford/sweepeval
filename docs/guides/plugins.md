# Plugin cookbook

Two extension points, both protocol-based and both registered by entry point.
Invariant I10: adding either touches no runner code.

Both snippets below are executed in CI, so they cannot rot.

## A custom scorer

Implement three attributes and two methods. `requires` names the capabilities
your scorer needs — if the target lacks one, the framework reports your family
as `SKIPPED` with the detector that ruled it out, rather than running it and
producing nonsense.

```python
from dataclasses import dataclass, field

from sweepeval.capabilities.detect import Capability
from sweepeval.schema.metric import MetricSpec
from sweepeval.schema.observation import Verdict


@dataclass
class BrandVoiceScorer:
    """Does the target use the words we told it to use?"""

    family: str = "brand_voice"
    version: int = 1
    requires: frozenset[Capability] = field(default_factory=frozenset)

    def metrics(self):
        return [
            MetricSpec(
                metric="brand_voice_rate",
                family="brand_voice",
                direction="maximize",
                unit="rate",
                cluster_key="probe",
            )
        ]

    def score(self, unit, calls, context):
        banned = {"synergy", "leverage", "circle back"}
        text = context.text.casefold()

        if not text.strip():
            # I5: never a silent zero. An empty answer is unscorable, and the
            # reason travels with it into the report.
            return [
                context.observation(
                    scorer=self.family, version=self.version,
                    metric="brand_voice_rate", family=self.family,
                    verdict=Verdict.UNSCORABLE,
                    reason="no text extracted from the final turn", unit=unit,
                )
            ]

        hits = [word for word in banned if word in text]
        return [
            context.observation(
                scorer=self.family, version=self.version,
                metric="brand_voice_rate", family=self.family,
                verdict=Verdict.FAIL if hits else Verdict.PASS,
                value=0.0 if hits else 1.0,
                reason=f"used {', '.join(sorted(hits))}" if hits else "clean",
                unit=unit,
            )
        ]
```

Register it in your own package's metadata:

```toml
[project.entry-points."sweepeval.scorers"]
brand_voice = "my_package.scorers:BrandVoiceScorer"
```

Or in-process, for a script:

```python
from sweepeval.scorers import register

register(BrandVoiceScorer())
```

### Three rules for a scorer

1. **Never return a silent zero.** If you cannot score, return `UNSCORABLE`
   with a reason. A zero is a measurement; an absence is not.
2. **Declare `requires` honestly.** A scorer that needs multi-turn state and
   does not say so will produce garbage against a stateless endpoint instead
   of being skipped.
3. **Bump `version` when you change what the rate counts.** It is a hard
   comparability key, so old and new runs will correctly refuse to be
   compared.

## A custom discovery shape

If your endpoint's request format is not one of the six built-ins, teach
discovery about it. Priorities are spaced by 10 so a third-party shape slots
between two built-ins without renumbering them.

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class AcmeCompletion:
    """POST {"query": "...", "opts": {...}} -> {"data": {"answer": "..."}}"""

    name: str = "acme.completion"
    priority: float = 15.0
    default_paths: tuple[str, ...] = ("/api/v2/complete",)

    def build(self, prompt: str, **params: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"query": prompt, "opts": {}}
        for key in ("temperature", "top_p", "model"):
            if params.get(key) is not None:
                body["opts"][key] = params[key]
        return body

    def build_multi_turn(
        self, turns: list[tuple[str, str]], **params: Any
    ) -> dict[str, Any]:
        # No history slot, so flatten. Discovery detects statelessness and the
        # multi-turn capability reports it, so nothing downstream pretends the
        # endpoint has a session.
        joined = "\n".join(f"{role}: {text}" for role, text in turns)
        return self.build(joined, **params)

    def prior_text_paths(self) -> tuple[str, ...]:
        # Tried before the blind walk. The nonce oracle is still authoritative.
        return ("$.data.answer", "$.data.text")
```

```toml
[project.entry-points."sweepeval.discovery_shapes"]
acme = "my_package.shapes:AcmeCompletion"
```

## A custom objective

Objectives are data, not code. Register one to put your scorer's metric on the
frontier:

```python
from sweepeval.schema.objective import REGISTRY, Objective

REGISTRY.register(
    Objective(
        id="brand_voice_rate",
        display_label="Brand voice",
        direction="maximize",
        family="brand_voice",
        cluster_key="probe",
        min_effect=0.05,
        min_effect_kind="absolute",
        default=False,
        weighting="unweighted mean over probes",
    )
)
```

`default=False` keeps it off the frontier until someone asks for it:

```bash
sweepeval report .sweepeval/runs/<id> --objectives security,brand_voice_rate
```

`min_effect` is the margin below which a difference is not worth calling a
difference. Setting it to zero makes every trivial difference significant
given enough runs, which is how a frontier ends up empty.
