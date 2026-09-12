"""Stage A: the free metadata sniff (spec §8.1).

Token-free and not charged against the POST budget. A single hit here often
collapses the ladder to one confirming request — and on a target that lists its
models, it also supplies the model axis the sweep will later need.

Nothing here is a prompt. These are GETs and an OPTIONS.

The sniff **authenticates when a key is supplied**. It did not, and the
consequence was severe: OpenAI requires auth on ``/v1/models``, so the sniff
got a 401, learned no models and suggested no shape, and the mutator that
later needed a model id had nothing to offer but a guess. Discovery against
the most common endpoint in the world failed on a metadata request that would
have succeeded with the key the user had already handed over.

Auth style is not known yet at this point -- that is what the POST ladder
resolves -- so the sniff tries the two header styles that cover essentially
every endpoint. These are GETs against metadata paths: they cost no tokens and
are not charged against the POST budget, so a second attempt is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

__all__ = ["SNIFF_PATHS", "SniffResult", "sniff"]

SNIFF_PATHS: tuple[str, ...] = (
    "/v1/models",
    "/models",
    "/openapi.json",
    "/.well-known/ai-plugin.json",
)

# Families identifiable from a metadata response alone.
_FAMILY_HINTS: tuple[tuple[str, str], ...] = (
    ("/v1/chat/completions", "openai.chat_completions"),
    ("/chat/completions", "openai.chat_completions"),
    ("/v1/messages", "anthropic.messages"),
    (":generateContent", "gemini.generate_content"),
)


@dataclass
class SniffResult:
    """What the free stage learned."""

    models: tuple[str, ...] = ()
    openapi_paths: tuple[str, ...] = ()
    suggested_shape: str | None = None
    suggested_paths: tuple[str, ...] = ()
    allow: tuple[str, ...] = ()
    reachable: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def informative(self) -> bool:
        return bool(self.models or self.openapi_paths or self.suggested_shape)


def _auth_attempts(key: str | None) -> tuple[dict[str, str], ...]:
    """Header sets to try, cheapest first: none, then the two common styles."""
    if not key:
        return ({},)
    return ({}, {"authorization": f"Bearer {key}"}, {"x-api-key": key})


async def sniff(
    client: httpx.AsyncClient, base: str, key: str | None = None
) -> SniffResult:
    """GET the metadata endpoints. Never raises; a sniff is best-effort."""
    result = SniffResult()
    evidence: dict[str, Any] = {}

    root = base.rstrip("/")
    origin = _origin(root)

    try:
        options = await client.options(root)
        result.reachable = True
        allow = options.headers.get("allow", "")
        result.allow = tuple(m.strip() for m in allow.split(",") if m.strip())
        evidence["OPTIONS"] = options.status_code
    except httpx.HTTPError as exc:
        evidence["OPTIONS"] = f"error: {type(exc).__name__}"

    for path in SNIFF_PATHS:
        url = f"{origin}{path}"
        response = None
        for headers in _auth_attempts(key):
            try:
                response = await client.get(url, headers=headers)
            except httpx.HTTPError as exc:
                evidence[path] = f"error: {type(exc).__name__}"
                response = None
                break
            evidence[path] = response.status_code
            # Only an auth rejection is worth another header style. A 404 is a
            # path that does not exist, and no credential fixes that.
            if response.status_code not in (401, 403):
                break

        if response is None or response.status_code != 200:
            continue
        result.reachable = True

        try:
            payload = response.json()
        except ValueError:
            continue

        if path in {"/v1/models", "/models"}:
            result.models = _models_from(payload)
            if result.models:
                evidence[f"{path}.models"] = list(result.models)
        elif path == "/openapi.json":
            paths = tuple(sorted((payload.get("paths") or {}).keys()))
            result.openapi_paths = paths
            if paths:
                evidence["openapi.paths"] = list(paths)

    shape, suggested = _suggest(result)
    result.suggested_shape = shape
    result.suggested_paths = suggested
    result.evidence = evidence
    return result


def _origin(url: str) -> str:
    parts = httpx.URL(url)
    return f"{parts.scheme}://{parts.netloc.decode()}"


def _models_from(payload: Any) -> tuple[str, ...]:
    if isinstance(payload, dict):
        data = payload.get("data") or payload.get("models")
        if isinstance(data, list):
            ids = [
                item.get("id") or item.get("name")
                for item in data
                if isinstance(item, dict)
            ]
            return tuple(sorted(str(i) for i in ids if i))
    if isinstance(payload, list):
        return tuple(sorted(str(i) for i in payload if isinstance(i, str)))
    return ()


def _suggest(result: SniffResult) -> tuple[str | None, tuple[str, ...]]:
    """Collapse the ladder when the metadata already names a family."""
    for path in result.openapi_paths:
        for needle, shape in _FAMILY_HINTS:
            if needle in path:
                return shape, (path,)

    # A /v1/models response is itself strong evidence of the OpenAI family.
    if result.models:
        return "openai.chat_completions", ("/v1/chat/completions",)

    return None, ()
