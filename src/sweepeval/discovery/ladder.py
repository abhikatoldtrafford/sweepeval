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

__all__ = ["LadderResult", "body_for_turns", "climb", "model_in_body"]


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

    pinned: dict[str, Any] = field(default_factory=dict)
    """Params every body built from this ladder carries unless overridden.

    Structural on purpose. ``--model`` first pinned the model on the probe
    plan alone, so the probes named it and capability detection -- five
    detectors and the sampling probe, all building their bodies straight off
    the ladder -- went on naming whichever model discovery had picked. The
    capability report would then describe a different model from the one the
    metrics came from, and the two would disagree exactly where it matters:
    a reasoning model rejects the sampling parameters a chat model accepts.

    Putting it here means every caller of :func:`body_for_turns` inherits it
    and there is no call site left to miss. A swept param still wins: the
    sweep sets it per config, and this is the run-wide floor.
    """

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
    sniffed = await sniff(client, url, key)

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

    # Values the free metadata sniff actually saw, for the mutator to use when
    # an error demands a field. `model` is the one that decides whether
    # discovery against OpenAI succeeds at all: it rejects a body with no
    # model, and a guessed name only turns that into "the model does not
    # exist". A model id from /v1/models is the endpoint's own answer.
    hints: dict[str, object] = {}
    if sniffed.models:
        hints["model"] = _preferred_model(sniffed.models)

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
                resolved=resolved, allow_mutation=allow_mutation, hints=hints,
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
                    resolved=resolved, hints=hints,
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


def _preferred_model(models: tuple[str, ...]) -> str:
    """Which discovered model to put in a probe body.

    Discovery has to answer with *one* model before the planner has run, and
    the choice is not neutral: a gateway lists embeddings, moderation and TTS
    models that will never answer a chat request, and reasoning models reject
    the sampling parameters later probes send. Prefer a cheap, plain chat
    model, then anything that is not obviously non-chat, then give up and take
    the first -- an inert prompt against the wrong model still produces an
    error message worth reading.
    """
    from sweepeval.execute.planner import filter_model_ids

    usable, _dropped = filter_model_ids(list(models), bound=len(models) or 1)
    if not usable:
        return models[0]

    for token in ("mini", "flash", "haiku", "small", "lite", "turbo"):
        for model in usable:
            if token in model.lower():
                return model
    return usable[0]


def body_for_turns(
    ladder: LadderResult, turns: list[tuple[str, str]], **params: Any
) -> dict[str, Any]:
    """Build a request body for a conversation, honouring mutations (§8.2).

    A shape that only reached a 200 through mutation cannot be rebuilt from the
    shape alone: the prompt lives in a field the shape does not know about — a
    renamed or added one — and ``shape.build_multi_turn`` would put the text
    somewhere the target never reads while leaving the old text in place. That
    produces a 200 carrying nothing, which is worse than a failure because it
    looks like success.

    With no mutations the shape builds the body directly, history and all.
    With mutations the conversation is flattened into whichever field carried
    the inert prompt, because a mutated body has no history slot to fill.

    **A purely additive mutation is the exception**, and it is the common case
    rather than a corner: OpenAI is discovered as ``openai.chat_completions +
    add:model``, because it rejects a body with no model. The shape's own
    structure is untouched there — only an extra top-level key was added — so
    flattening would throw away the history slot the shape does have. Measured
    against the real endpoint, that collapsed every multi-turn probe into one
    message and silently dropped the swept ``model`` and ``temperature``, so
    the context family measured nothing and two axes of the sweep did not
    exist. The shape builds the body; the added keys are merged back; swept
    params win over the discovered value.
    """
    if ladder.pinned:
        params = {**ladder.pinned, **params}
    if not ladder.mutations:
        return ladder.shape.build_multi_turn(turns, **params)

    additions = _additive_only(ladder)
    if additions is not None:
        # Additions first, so a swept parameter wins over the value discovery
        # happened to probe with. The other order pins `model` to whichever id
        # the metadata sniff picked, and the model axis silently does nothing.
        return {**additions, **ladder.shape.build_multi_turn(turns, **params)}

    flattened = "\n".join(f"{role}: {text}" for role, text in turns)
    if "__raw__" in ladder.body:
        return {"__raw__": flattened}

    substituted = _substitute_prompt(ladder.body, INERT_PROMPT, flattened)
    if substituted != ladder.body:
        return substituted
    return ladder.shape.build_multi_turn(turns, **params)


def model_in_body(ladder: LadderResult, **params: Any) -> str | None:
    """The model id the next request will actually carry, or ``None``.

    Read out of a body built by :func:`body_for_turns` rather than off the
    params or the discovered body, because those two disagree and only the
    built body is what goes on the wire: discovery reaches OpenAI as
    ``openai.chat_completions + add:model`` and the added value is *overridden*
    by a swept ``model``, while for a shape that puts the model in the URL
    there is no body field to override at all.

    ``None`` is a real answer and means "this request carries no model field".
    A shape like ``gemini.generate_content`` names the model in the path, and
    ``raw.text`` has no notion of one; asserting a model for either would be
    provenance that is not true.
    """
    body = body_for_turns(ladder, [("user", INERT_PROMPT)], **params)
    value = body.get("model") if isinstance(body, dict) else None
    return value if isinstance(value, str) and value else None


def _additive_only(ladder: LadderResult) -> dict[str, Any] | None:
    """The keys a mutation *added*, if it changed nothing the shape built.

    Returns ``None`` when the mutation touched the shape's own structure — a
    rename, a nest, an unwrap, a coercion — because there the shape can no
    longer build a body the target accepts and flattening is the honest
    fallback.

    Only top-level keys the shape did not produce count as additions, and
    every key the shape *did* produce must still be present and unchanged.
    Anything less strict would let a rename look additive.
    """
    built = ladder.shape.build(INERT_PROMPT)
    if not isinstance(ladder.body, dict) or "__raw__" in ladder.body:
        return None

    for key, value in built.items():
        if key not in ladder.body or ladder.body[key] != value:
            return None

    additions = {k: v for k, v in ladder.body.items() if k not in built}
    return additions or None


def _substitute_prompt(node: Any, old: str, new: str) -> Any:
    if isinstance(node, str):
        return new if old in node else node
    if isinstance(node, dict):
        return {k: _substitute_prompt(v, old, new) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute_prompt(v, old, new) for v in node]
    return node


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
    hints: dict[str, object] | None = None,
) -> LadderResult | None:
    """Try one shape, walking the auth ladder and then mutating on rejection."""
    body = shape.build(INERT_PROMPT)
    cell = resolved if resolved is not None else [None]

    candidates = (cell[0],) if cell[0] is not None else auth_ladder(key)

    for auth in candidates:
        assert auth is not None
        result = await _post_and_mutate(
            client, url, shape, auth, key, body, budget, stage=stage,
            allow_mutation=allow_mutation, hints=hints,
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
    hints: dict[str, object] | None = None,
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

    for mutation, mutated in mutations_for(body, raw, INERT_PROMPT, hints):
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
            hints=hints,
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
