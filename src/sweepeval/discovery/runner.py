"""Discovery orchestration (spec §8). Produces the annotated config.

Ties the stages together: sniff, ladder, nonce oracle, fallback walk, target
type. The output is the payload :mod:`sweepeval.discovery.emit` writes.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

import httpx

from sweepeval.discovery.budget import INERT_PROMPT, DiscoveryBudget
from sweepeval.discovery.extract import (
    ExtractionResult,
    infer_from_nonce,
    infer_from_walk,
)
from sweepeval.discovery.ladder import LadderResult, climb
from sweepeval.schema.versions import SCHEMA_VERSION, TOOL_VERSION
from sweepeval.store.redaction import Redactor

__all__ = ["DiscoveryOutcome", "discover_target"]

# Two probes with deliberately different, non-trivial content: the fallback
# walk needs input variation to tell an answer from a constant.
_WALK_PROBES = (
    "Name three primary colours, separated by commas.",
    "In one short sentence, describe what a compiler does.",
)


@dataclass
class DiscoveryOutcome:
    ladder: LadderResult
    extraction: ExtractionResult
    target_type: str
    target_type_confidence: str
    target_type_evidence: dict[str, Any]
    budget: DiscoveryBudget
    payload: dict[str, Any] = field(default_factory=dict)


async def discover_target(
    client: httpx.AsyncClient,
    url: str,
    key: str | None = None,
    *,
    budget: DiscoveryBudget | None = None,
    seed: int | None = None,
) -> DiscoveryOutcome:
    """Run Phase 0 and build the annotated config payload."""
    budget = budget or DiscoveryBudget()
    redactor = Redactor(secrets=[key] if key else [])

    ladder = await climb(client, url, key, budget)
    extraction = await _infer_extraction(client, ladder, key, budget, seed)
    target_type, tt_confidence, tt_evidence = _infer_target_type(ladder, extraction)

    payload = _build_payload(
        url=url,
        ladder=ladder,
        extraction=extraction,
        target_type=target_type,
        tt_confidence=tt_confidence,
        tt_evidence=tt_evidence,
        budget=budget,
        redactor=redactor,
    )

    return DiscoveryOutcome(
        ladder=ladder,
        extraction=extraction,
        target_type=target_type,
        target_type_confidence=tt_confidence,
        target_type_evidence=tt_evidence,
        budget=budget,
        payload=payload,
    )


async def _infer_extraction(
    client: httpx.AsyncClient,
    ladder: LadderResult,
    key: str | None,
    budget: DiscoveryBudget,
    seed: int | None,
) -> ExtractionResult:
    """Nonce oracle first; priors plus the blind walk only if it fails (§8.5)."""
    from sweepeval.discovery.auth import apply_auth

    headers, params = apply_auth(ladder.auth, key)
    nonce = _nonce(seed)

    oracle_body = _rebuild(
        ladder, f"Reply with exactly the following and nothing else: {nonce}"
    )
    payload = await _post_json(client, ladder.path, oracle_body, headers, params, budget)

    if payload is not None:
        result = infer_from_nonce(payload, nonce)
        if result.ok:
            # Cross-check against the family prior. Agreement raises
            # confidence; disagreement is recorded as an assumption to correct.
            priors = ladder.shape.prior_text_paths()
            result.evidence["prior_agrees"] = result.path in priors
            result.evidence["family_priors"] = list(priors)
            return result

    # Fallback: two varied probes, then score paths.
    payloads: list[Any] = []
    prompts: list[str] = []
    for probe in _WALK_PROBES:
        if budget.exhausted():
            break
        body = _rebuild(ladder, probe)
        got = await _post_json(client, ladder.path, body, headers, params, budget)
        if got is not None:
            payloads.append(got)
            prompts.append(probe)

    if ladder.response_json is not None:
        payloads.append(ladder.response_json)
        prompts.append(INERT_PROMPT)

    fallback = infer_from_walk(
        payloads, prompts, priors=ladder.shape.prior_text_paths()
    )
    fallback.evidence["oracle_failed"] = True
    return fallback


def _rebuild(ladder: LadderResult, prompt: str) -> dict[str, Any]:
    """Rebuild the winning body with new prompt text.

    Substitutes into the *winning* body rather than rebuilding from the shape.
    A body that reached a 200 through mutation may carry the prompt in a field
    the shape does not know about — a renamed or nested one — and rebuilding
    from the shape would put the new text somewhere the target never reads,
    leaving the old text in place. That silently defeats the nonce oracle,
    because the response then contains no nonce and extraction falls back for
    no reason.
    """
    if "__raw__" in ladder.body:
        return {"__raw__": prompt}

    substituted = _substitute(ladder.body, INERT_PROMPT, prompt)
    if substituted != ladder.body:
        return substituted  # type: ignore[return-value]

    # The winning body did not visibly carry the inert prompt (an unusual
    # shape); fall back to asking the shape to build a fresh one.
    return ladder.shape.build(prompt)


def _substitute(node: Any, old: str, new: str) -> Any:
    if isinstance(node, str):
        return new if old in node else node
    if isinstance(node, dict):
        return {k: _substitute(v, old, new) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, old, new) for v in node]
    return node


async def _post_json(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    params: dict[str, Any],
    budget: DiscoveryBudget,
) -> Any:
    from sweepeval.discovery.budget import Attempt
    from sweepeval.discovery.ladder import _post

    if budget.exhausted():
        return None

    status, content_type, payload, raw = await _post(client, url, body, headers, params)
    budget.record(
        Attempt(
            stage="extract",
            shape="-",
            path=httpx.URL(url).path,
            mutation=None,
            status=status,
            content_type=content_type,
            error_excerpt="" if status < 400 else raw[:120].decode("utf-8", "replace"),
        ),
        tokens=40,
    )
    return payload if 200 <= status < 300 else None


def _nonce(seed: int | None) -> str:
    if seed is None:
        return secrets.token_hex(5).upper()
    import random

    rng = random.Random(seed)
    return "".join(rng.choice("0123456789ABCDEF") for _ in range(10))


def _infer_target_type(
    ladder: LadderResult, extraction: ExtractionResult
) -> tuple[str, str, dict[str, Any]]:
    """BARE MODEL vs AGENT SYSTEM (§9).

    **Judgement call, flagged by the plan:** the spec describes this as "from
    tool/retrieval/latency evidence" with no threshold. The rule used here is
    structural and recorded as a low-confidence assumption, because a wrong
    guess blocks a legitimate comparison (I6) rather than corrupting a metric.

    A target is an AGENT SYSTEM when its response carries structure a bare
    completion endpoint would not produce: retrieved documents, tool calls, or
    a non-standard envelope that no family prior matches.
    """
    payload = ladder.response_json
    evidence: dict[str, Any] = {}

    has_tools = _has_key(payload, {"tool_calls", "tool_use", "function_call"})
    has_docs = _looks_like_documents(payload)
    off_prior = extraction.ok and extraction.path not in ladder.shape.prior_text_paths()

    evidence["tool_structure"] = has_tools
    evidence["retrieved_documents"] = has_docs
    evidence["extraction_path_off_family_prior"] = off_prior

    if has_tools or has_docs:
        return "AGENT_SYSTEM", "medium", evidence
    if off_prior:
        return "AGENT_SYSTEM", "low", evidence
    return "BARE_MODEL", "low", evidence


def _has_key(payload: Any, names: set[str]) -> bool:
    if isinstance(payload, dict):
        if names & set(payload):
            return True
        return any(_has_key(v, names) for v in payload.values())
    if isinstance(payload, list):
        return any(_has_key(v, names) for v in payload)
    return False


def _looks_like_documents(payload: Any) -> bool:
    """An array of objects carrying id/score/text shapes (§8.5)."""
    if isinstance(payload, dict):
        return any(_looks_like_documents(v) for v in payload.values())
    if isinstance(payload, list) and len(payload) >= 2:
        objects = [v for v in payload if isinstance(v, dict)]
        if len(objects) == len(payload):
            keys = set().union(*(set(o) for o in objects))
            if len({"score", "text"} & keys) == 2 or len({"id", "text"} & keys) == 2:
                return True
    return False


def _build_payload(
    *,
    url: str,
    ladder: LadderResult,
    extraction: ExtractionResult,
    target_type: str,
    tt_confidence: str,
    tt_evidence: dict[str, Any],
    budget: DiscoveryBudget,
    redactor: Redactor,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "target": {
            "url": redactor.url(ladder.path),
            "endpoint_fingerprint": redactor.endpoint_fingerprint(ladder.path),
            "shape": ladder.shape.name,
            "shape_method": (
                f"stage {ladder.stage} ladder"
                + (f" + mutations {list(ladder.mutations)}" if ladder.mutations else "")
            ),
            "auth": ladder.auth.name,
            "auth_method": "first success of the four-rung ladder (§8.4)",
            "key_in_url": ladder.puts_key_in_url,
            "target_type": target_type,
            "target_type_confidence": tt_confidence,
            "target_type_evidence": tt_evidence,
        },
        "extraction": {
            "text_path": extraction.path,
            "confidence": extraction.confidence,
            "method": extraction.method,
            "evidence": extraction.evidence,
        },
        "discovery": {
            "posts": budget.posts,
            "models_seen": list(ladder.sniff.models),
            "sniff": ladder.sniff.evidence,
            "transcript": [a.render().strip() for a in budget.attempts],
        },
    }
