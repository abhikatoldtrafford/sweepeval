"""The discovery ladder (spec §8.1, §8.2, §8.3). I8 load-bearing.

Three stages:

* **A** — the free metadata sniff (:mod:`sweepeval.discovery.sniff`).
* **B** — the six-shape ladder against the exact URL, in prior order, with
  error-guided mutation on each structural rejection.
* **C** — path completion, *only* when every shape returned 404/405, meaning
  the URL is a base rather than an endpoint.

Stage C is gated deliberately: the tool should not POST to a URL the user did
not name unless the evidence says the one they named is not an endpoint at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from sweepeval.discovery.auth import AuthMethod, apply_auth, auth_ladder
from sweepeval.discovery.budget import (
    INERT_PROMPT,
    Attempt,
    DiscoveryAborted,
    DiscoveryBudget,
)
from sweepeval.discovery.mutate import (
    MAX_MUTATION_ATTEMPTS,
    MAX_MUTATION_DEPTH,
    mutations_for,
)
from sweepeval.discovery.shapes import (
    Shape,
    builtin,  # noqa: F401  (registers built-ins)
    ladder_order,
)
from sweepeval.discovery.sniff import SniffResult, sniff

__all__ = ["LadderResult", "climb"]


@dataclass
class LadderResult:
    """The identified shape, path, auth method and the evidence for each."""

    shape: Shape
    path: str
    auth: AuthMethod
    body: dict[str, Any]
    response_json: Any
    sniff: SniffResult
    stage: str
    mutations: tuple[str, ...] = ()
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def puts_key_in_url(self) -> bool:
        return self.auth.puts_key_in_url


async def climb(
    client: httpx.AsyncClient,
    url: str,
    key: str | None,
    budget: DiscoveryBudget | None = None,
) -> LadderResult:
    """Identify the request shape, or abort with a full diagnostic."""
    budget = budget or DiscoveryBudget()
    sniffed = await sniff(client, url)

    shapes = list(ladder_order())
    if sniffed.suggested_shape:
        # Collapse the ladder: try the sniffed family first, keep the rest as
        # fallback in case the metadata was a proxy's rather than the target's.
        shapes.sort(key=lambda s: (s.name != sniffed.suggested_shape, s.priority))

    all_not_found = True
    # Auth is a property of the endpoint, not of the shape. Resolving it once
    # and reusing it is what keeps discovery inside 25 requests: re-walking a
    # four-rung ladder for each of six shapes, each of which may then mutate,
    # exhausts the budget before the correct shape is ever reached.
    resolved: list[AuthMethod | None] = [None]

    # --- stage B: the exact URL -------------------------------------------
    # Two passes. Every shape is tried unmutated before any shape is mutated,
    # because a mutated near-miss can satisfy a different family's endpoint:
    # an OpenAI body plus `add:max_tokens` is a valid Anthropic request, and
    # a single pass would label an Anthropic endpoint as OpenAI and then use
    # the wrong response priors and the wrong multi-turn body.
    for allow_mutation in (False, True):
        for shape in shapes:
            outcome = await _try_shape(
                client, url, shape, key, budget, stage="B",
                resolved=resolved, allow_mutation=allow_mutation,
            )
            if outcome is not None:
                outcome.sniff = sniffed
                return outcome
            if not _last_was_not_found(budget):
                all_not_found = False
            if budget.exhausted():
                raise budget.abort(url)

    # --- stage C: path completion ------------------------------------------
    if all_not_found:
        origin = _origin(url)
        for shape in shapes:
            for suffix in _candidate_paths(shape, sniffed):
                candidate = f"{origin}{suffix}"
                if candidate.rstrip("/") == url.rstrip("/"):
                    continue
                outcome = await _try_shape(
                    client, candidate, shape, key, budget, stage="C",
                    resolved=resolved,
                )
                if outcome is not None:
                    outcome.sniff = sniffed
                    return outcome
                if budget.exhausted():
                    raise budget.abort(url)

    raise budget.abort(
        url,
        reason=(
            "no candidate shape produced a usable response"
            if not budget.exhausted()
            else None
        ),
    )


def _candidate_paths(shape: Shape, sniffed: SniffResult) -> tuple[str, ...]:
    paths = tuple(p for p in shape.default_paths if "{" not in p)
    if sniffed.suggested_paths:
        return tuple(dict.fromkeys(sniffed.suggested_paths + paths))
    return paths


def _origin(url: str) -> str:
    parts = httpx.URL(url)
    return f"{parts.scheme}://{parts.netloc.decode()}"


def _last_was_not_found(budget: DiscoveryBudget) -> bool:
    return bool(budget.attempts) and budget.attempts[-1].status in (404, 405)


async def _try_shape(
    client: httpx.AsyncClient,
    url: str,
    shape: Shape,
    key: str | None,
    budget: DiscoveryBudget,
    *,
    stage: str,
    resolved: list[AuthMethod | None] | None = None,
    allow_mutation: bool = True,
) -> LadderResult | None:
    """Try one shape, walking the auth ladder and then mutating on rejection."""
    body = shape.build(INERT_PROMPT)
    cell = resolved if resolved is not None else [None]

    candidates = (cell[0],) if cell[0] is not None else auth_ladder(key)

    for auth in candidates:
        assert auth is not None
        result = await _post_and_mutate(
            client, url, shape, auth, key, body, budget, stage=stage,
            allow_mutation=allow_mutation,
        )
        if result is not None:
            cell[0] = auth
            return result
        if budget.exhausted():
            return None
        # A 401 means the shape may be right and the auth wrong: keep walking
        # the ladder. Anything else means auth was accepted, so this rung is
        # the endpoint's method — remember it and stop re-discovering it.
        if not budget.attempts or budget.attempts[-1].status != 401:
            cell[0] = auth
            return None
    return None


async def _post_and_mutate(
    client: httpx.AsyncClient,
    url: str,
    shape: Shape,
    auth: AuthMethod,
    key: str | None,
    body: dict[str, Any],
    budget: DiscoveryBudget,
    *,
    stage: str,
    depth: int = 0,
    mutations: tuple[str, ...] = (),
    spent: list[int] | None = None,
    allow_mutation: bool = True,
) -> LadderResult | None:
    if budget.exhausted():
        return None
    # One attempt budget for the whole mutation tree of this shape, not one
    # per level: per-level would allow MAX**DEPTH requests.
    spent = spent if spent is not None else [0]

    headers, params = apply_auth(auth, key)
    status, content_type, payload, raw = await _post(
        client, url, body, headers, params
    )

    budget.record(
        Attempt(
            stage=stage,
            shape=shape.name,
            path=httpx.URL(url).path,
            mutation=mutations[-1] if mutations else None,
            status=status,
            content_type=content_type,
            error_excerpt=_excerpt(raw) if status >= 400 else "",
            note=f"auth={auth.name}",
        ),
        tokens=len(INERT_PROMPT) // 4,
    )

    if 200 <= status < 300 and payload is not None:
        return LadderResult(
            shape=shape,
            path=url,
            auth=auth,
            body=body,
            response_json=payload,
            sniff=SniffResult(),
            stage=stage,
            mutations=mutations,
        )

    # Only a structural rejection is worth mutating along. A 401 is an auth
    # problem and a 404 is a path problem; neither is fixed by renaming a field.
    if not allow_mutation or status not in (400, 422) or depth >= MAX_MUTATION_DEPTH:
        return None

    for mutation, mutated in mutations_for(body, raw, INERT_PROMPT):
        if spent[0] >= MAX_MUTATION_ATTEMPTS or budget.exhausted():
            break
        spent[0] += 1
        result = await _post_and_mutate(
            client,
            url,
            shape,
            auth,
            key,
            mutated,
            budget,
            stage=stage,
            depth=depth + 1,
            mutations=(*mutations, mutation.label()),
            spent=spent,
            allow_mutation=True,
        )
        if result is not None:
            return result
    return None


async def _post(
    client: httpx.AsyncClient,
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    params: dict[str, Any],
) -> tuple[int, str, Any, bytes]:
    try:
        if "__raw__" in body:
            response = await client.post(
                url,
                content=str(body["__raw__"]).encode(),
                headers={**headers, "content-type": "text/plain"},
                params=params,
            )
        else:
            response = await client.post(url, json=body, headers=headers, params=params)
    except httpx.HTTPError as exc:
        return 0, "", None, str(exc).encode()

    raw = response.content
    content_type = response.headers.get("content-type", "")
    try:
        payload = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        payload = None
    return response.status_code, content_type, payload, raw


def _excerpt(raw: bytes, limit: int = 160) -> str:
    text = raw.decode("utf-8", errors="replace").strip().replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


__all__ += ["DiscoveryAborted"]
