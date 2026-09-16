"""The async HTTP client (spec §7, D29).

The only place in the package that performs a request — enforced by the
layering contract. Everything it knows about *when* to send is delegated to the
:class:`~sweepeval.http.governor.Governor`; everything it knows about *what*
came back is recorded as a :class:`~sweepeval.schema.call.Call` row.

Streaming is used whenever the target supports it, so first-token latency is
measured on every probe. Where the shape allows, the request also asks for
usage in the stream — most providers omit the usage block when streaming
unless asked, and without it nearly every run would silently drop from
``MEASURED`` to ``ESTIMATED`` cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from sweepeval.http.errors import classify_exception, classify_status
from sweepeval.http.governor import Governor, parse_retry_after
from sweepeval.http.streaming import decode_chunked_json, decode_sse, reassemble
from sweepeval.schema.call import (
    Call,
    CallRole,
    ErrorClass,
    ExtractionPart,
    RefusalPart,
    RequestPart,
    ResponsePart,
    TimingPart,
    TokensPart,
)
from sweepeval.schema.hashing import param_hash, sha256_hex

__all__ = ["CallResult", "TransportClient", "supports_usage_in_stream"]

# Judgement call flagged by the plan: the spec does not enumerate which shapes
# support usage-in-stream. OpenAI does via stream_options; every other shape is
# treated as unsupported and that is recorded in the manifest.
_USAGE_IN_STREAM_SHAPES = frozenset({"openai.chat_completions"})


def supports_usage_in_stream(shape: str) -> bool:
    return shape in _USAGE_IN_STREAM_SHAPES


@dataclass
class CallResult:
    """One completed attempt: the row, plus the content the row only hashes."""

    call: Call
    text: str
    raw_body: bytes
    events: list[tuple[str, dict[str, Any] | None]]
    retry_after: float | None = None
    """Parsed ``Retry-After``. Carried here rather than on ``Call`` because it
    is a transport hint for the next attempt, not a property of this one."""


class TransportClient:
    """Async wrapper around ``httpx.AsyncClient``."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        governor: Governor,
        *,
        run_id: str,
        shape: str = "unknown",
    ) -> None:
        self._client = client
        self._governor = governor
        self._run_id = run_id
        self._shape = shape

    @property
    def governor(self) -> Governor:
        """The rate decisions this client obeys.

        Exposed because the degradation ramp has to widen the in-flight limit
        for its own probes and nothing else, and it does that by asking the
        governor rather than by dispatching around it -- every rate decision
        stays in one place (§7).
        """
        return self._governor

    async def call(
        self,
        url: str,
        body: dict[str, Any],
        *,
        config_id: str,
        unit_id: str,
        run_idx: int = 0,
        turn_idx: int = 0,
        role: CallRole = "target",
        headers: dict[str, str] | None = None,
        stream: bool = False,
        params: dict[str, Any] | None = None,
        in_flight: int = 1,
    ) -> list[CallResult]:
        """Send one logical call, retrying per policy.

        Returns one :class:`CallResult` per *attempt*, so ``calls.jsonl`` holds
        a row for every request actually sent — a run that succeeded only after
        three retries looks different from one that succeeded immediately, and
        the operational metrics need to be able to tell them apart.
        """
        results: list[CallResult] = []

        for attempt in range(1, self._governor.max_attempts + 1):
            self._governor.check_budget()

            queue_start = time.perf_counter()
            async with self._governor.semaphore:
                queue_ms = (time.perf_counter() - queue_start) * 1000.0
                result = await self._attempt(
                    url,
                    body,
                    headers=headers,
                    params=params,
                    stream=stream,
                    attempt=attempt,
                    queue_ms=queue_ms,
                    config_id=config_id,
                    unit_id=unit_id,
                    run_idx=run_idx,
                    turn_idx=turn_idx,
                    role=role,
                    in_flight=in_flight,
                )

            results.append(result)
            self._governor.record_spend(
                requests=1, tokens=(result.call.tokens.out or 0)
            )
            error_class = result.call.response.error_class
            self._governor.record_outcome(config_id, error_class)

            if not self._governor.should_retry(error_class, attempt):
                break

            await self._sleep(self._governor.delay_for(attempt, result.retry_after))

        return results

    @staticmethod
    async def _sleep(seconds: float) -> None:
        import asyncio

        if seconds > 0:
            await asyncio.sleep(seconds)

    async def _attempt(
        self,
        url: str,
        body: dict[str, Any],
        *,
        headers: dict[str, str] | None,
        params: dict[str, Any] | None,
        stream: bool,
        attempt: int,
        queue_ms: float,
        in_flight: int = 1,
        config_id: str,
        unit_id: str,
        run_idx: int,
        turn_idx: int,
        role: CallRole,
    ) -> CallResult:
        payload = dict(body)
        if stream:
            payload["stream"] = True
            if supports_usage_in_stream(self._shape):
                payload["stream_options"] = {"include_usage": True}

        request_part = RequestPart(
            shape=self._shape,
            params_hash=param_hash(
                {k: v for k, v in payload.items() if k not in {"messages", "prompt"}}
            ),
            body_sha256=sha256_hex(repr(sorted(payload.items())).encode()),
            bytes=len(repr(payload)),
        )

        started = time.perf_counter()
        ttft_ms: float | None = None
        events: list[tuple[str, dict[str, Any] | None]] = []
        raw = b""
        retry_after_header: str | None = None

        try:
            if stream:
                async with self._client.stream(
                    "POST", url, json=payload, headers=headers, params=params
                ) as response:
                    status = response.status_code
                    retry_after_header = response.headers.get("retry-after")
                    content_type = response.headers.get("content-type", "")
                    chunks: list[bytes] = []
                    async for chunk in response.aiter_bytes():
                        if ttft_ms is None and chunk.strip():
                            ttft_ms = (time.perf_counter() - started) * 1000.0
                        chunks.append(chunk)
                    raw = b"".join(chunks)
                    decoder = (
                        decode_sse
                        if "event-stream" in content_type
                        else decode_chunked_json
                    )
                    events = list(decoder(chunks))
                    text = reassemble(events)
            else:
                response = await self._client.post(
                    url, json=payload, headers=headers, params=params
                )
                status = response.status_code
                retry_after_header = response.headers.get("retry-after")
                raw = response.content
                ttft_ms = (time.perf_counter() - started) * 1000.0
                text = ""
            total_ms = (time.perf_counter() - started) * 1000.0
            error_class = classify_status(status)
        except Exception as exc:
            total_ms = (time.perf_counter() - started) * 1000.0
            status = 0
            text = ""
            error_class = classify_exception(exc)

        call = Call(
            ts=datetime.now(timezone.utc).isoformat(),
            run_id=self._run_id,
            config_id=config_id,
            unit_id=unit_id,
            run_idx=run_idx,
            turn_idx=turn_idx,
            attempt=attempt,
            role=role,
            in_flight=in_flight,
            request=request_part,
            response=ResponsePart(
                status=status,
                error_class=error_class,
                streamed=stream,
                bytes=len(raw),
                body_sha256=sha256_hex(raw) if raw else None,
                error_excerpt=_excerpt(raw, error_class),
            ),
            timing=TimingPart(queue_ms=queue_ms, ttft_ms=ttft_ms, total_ms=total_ms),
            tokens=self._tokens(raw, text, events),
            extraction=ExtractionPart(path=None, ok=False),
            refusal=RefusalPart(),
        )

        return CallResult(
            call=call,
            text=text,
            raw_body=raw,
            events=events,
            retry_after=parse_retry_after(retry_after_header),
        )

    @staticmethod
    def _tokens(
        raw: bytes, text: str, events: list[tuple[str, dict[str, Any] | None]]
    ) -> TokensPart:
        import json as _json

        usage: dict[str, Any] | None = None
        try:
            parsed = _json.loads(raw) if raw else None
            if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict):
                usage = parsed["usage"]
        except _json.JSONDecodeError:
            pass

        if usage is None:
            for _, event in events:
                if isinstance(event, dict) and isinstance(event.get("usage"), dict):
                    usage = event["usage"]

        if usage is not None:
            return TokensPart(
                in_=usage.get("prompt_tokens") or usage.get("input_tokens"),
                out=usage.get("completion_tokens") or usage.get("output_tokens"),
                reasoning=usage.get("reasoning_tokens"),
                source="MEASURED",
            )

        # §12.4 tier 2: disclosed chars/4 heuristic, labelled ESTIMATED.
        return TokensPart(
            out=max(1, len(text) // 4) if text else None, source="ESTIMATED"
        )


ERROR_EXCERPT_CHARS = 400
"""How much of a failed body to keep on the row.

Enough to carry a provider's message -- "maximum context length is 128000
tokens" and the like -- and short enough that a run of 5xx does not turn
calls.jsonl into a copy of the response bodies. The full body is in the blob
store either way; this is the part a scorer can read without going there.
"""


def _excerpt(raw: bytes, error_class: ErrorClass) -> str | None:
    """The head of a failed response body. ``None`` when the call succeeded."""
    if error_class is ErrorClass.ok or not raw:
        return None
    return raw[: ERROR_EXCERPT_CHARS * 4].decode("utf-8", errors="replace")[
        :ERROR_EXCERPT_CHARS
    ]
