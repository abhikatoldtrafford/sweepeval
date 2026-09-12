"""Scenario-driven mock endpoint (spec §17).

A minimal ASGI app with no web-framework dependency. Tests mount it through
``httpx.ASGITransport``, so the whole suite runs with zero sockets and zero
tokens; ``sweepeval mock serve`` exposes the same app on a real port.

The app is stateful on purpose. Context drop at a chosen depth, response
caching, and rate limiting after N requests cannot be expressed by replaying a
recorded tape, and those are exactly the behaviours that separate configs.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from sweepeval.mock.scenario import Scenario

__all__ = ["MockApp"]

_INERT_REPLY = "OK"


class MockApp:
    """ASGI application whose behaviour comes from a :class:`Scenario`."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.request_count = 0
        self._nonce_counter = 0
        self._cache: dict[str, str] = {}
        self._call_log: list[dict[str, Any]] = []

    # --- ASGI --------------------------------------------------------------

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":  # pragma: no cover - lifespan etc.
            return

        body = await self._read_body(receive)
        status, headers, payload = self._handle(scope, body)

        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (k.encode("latin-1"), v.encode("latin-1")) for k, v in headers
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload})

    @staticmethod
    async def _read_body(receive: Any) -> bytes:
        body = b""
        while True:
            message = await receive()
            body += message.get("body", b"")
            if not message.get("more_body", False):
                return body

    # --- routing -----------------------------------------------------------

    def _handle(
        self, scope: dict[str, Any], body: bytes
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        method: str = scope["method"]
        path: str = scope["path"]
        query: str = scope.get("query_string", b"").decode()
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        scenario = self.scenario

        if method == "OPTIONS":
            return 200, [("allow", "POST, OPTIONS")], b""

        if method == "GET":
            return self._handle_get(path)

        if method != "POST":
            return 405, [("content-type", "application/json")], b'{"error":"method"}'

        if path not in scenario.paths:
            return (
                404,
                [("content-type", "application/json")],
                json.dumps({"error": {"message": f"no such path {path}"}}).encode(),
            )

        auth_error = self._check_auth(headers, query)
        if auth_error is not None:
            return auth_error

        self.request_count += 1

        if scenario.rate_limit_after and self.request_count > scenario.rate_limit_after:
            hdrs = [("content-type", "application/json")]
            if scenario.retry_after_s is not None:
                hdrs.append(("retry-after", str(scenario.retry_after_s)))
            return 429, hdrs, b'{"error":{"message":"rate limited"}}'

        if scenario.latency_ms:
            time.sleep(scenario.latency_ms / 1000.0)

        try:
            request = json.loads(body) if body else {}
        except json.JSONDecodeError:
            return (
                400,
                [("content-type", "application/json")],
                json.dumps(
                    {"error": {"message": "invalid JSON body", "param": None}}
                ).encode(),
            )

        shape_error = self._check_shape(request)
        if shape_error is not None:
            return shape_error

        prompt = self._extract_prompt(request)
        if (
            scenario.max_input_chars is not None
            and len(prompt) > scenario.max_input_chars
        ):
            return (
                400,
                [("content-type", "application/json")],
                json.dumps(
                    {
                        "error": {
                            "message": (
                                "maximum context length is "
                                f"{scenario.max_input_chars} characters"
                            ),
                            "code": "context_length_exceeded",
                        }
                    }
                ).encode(),
            )

        text = self._reply_text(request, prompt)

        if scenario.malformed_json:
            return 200, [("content-type", "application/json")], b'{"choices": [{"mess'

        return (
            200,
            [("content-type", "application/json")],
            json.dumps(self._envelope(text, prompt, request)).encode(),
        )

    def _handle_get(self, path: str) -> tuple[int, list[tuple[str, str]], bytes]:
        scenario = self.scenario
        json_hdr = [("content-type", "application/json")]

        if path == "/v1/models":
            if not scenario.expose_models:
                return 404, json_hdr, b'{"error":"not found"}'
            payload = {
                "object": "list",
                "data": [{"id": m, "object": "model"} for m in scenario.expose_models],
            }
            return 200, json_hdr, json.dumps(payload).encode()

        if path == "/openapi.json":
            if not scenario.expose_openapi:
                return 404, json_hdr, b'{"error":"not found"}'
            payload = {
                "openapi": "3.0.0",
                "paths": {p: {"post": {}} for p in scenario.paths},
            }
            return 200, json_hdr, json.dumps(payload).encode()

        return 404, json_hdr, b'{"error":"not found"}'

    # --- auth --------------------------------------------------------------

    def _check_auth(
        self, headers: dict[str, str], query: str
    ) -> tuple[int, list[tuple[str, str]], bytes] | None:
        scenario = self.scenario
        if scenario.auth == "none" or scenario.expected_key is None:
            return None

        supplied: str | None = None
        if scenario.auth == "bearer":
            value = headers.get("authorization", "")
            supplied = value[7:] if value.lower().startswith("bearer ") else None
        elif scenario.auth == "x-api-key":
            supplied = headers.get("x-api-key")
        elif scenario.auth == "api-key":
            supplied = headers.get("api-key")
        elif scenario.auth == "query":
            for pair in query.split("&"):
                key, _, value = pair.partition("=")
                if key in {"api_key", "key"}:
                    supplied = value

        if supplied != scenario.expected_key:
            return (
                401,
                [("content-type", "application/json")],
                json.dumps(
                    {"error": {"message": "invalid authentication", "type": "auth"}}
                ).encode(),
            )
        return None

    # --- shape -------------------------------------------------------------

    def _check_shape(
        self, request: dict[str, Any]
    ) -> tuple[int, list[tuple[str, str]], bytes] | None:
        """Reject bodies that do not match this scenario's shape.

        The error body names the field it wanted, which is the signal
        error-guided mutation searches on (§8.2).
        """
        required = {
            "openai": "messages",
            "anthropic": "messages",
            "gemini": "contents",
            "prompt": "prompt",
            "input": "input",
            "raw": None,
            "weird": "payload",
        }[self.scenario.shape]

        if required is None or required in request:
            return None

        return (
            400,
            [("content-type", "application/json")],
            json.dumps(
                {
                    "error": {
                        "message": f"missing required field: '{required}'",
                        "param": required,
                        "type": "invalid_request_error",
                    }
                }
            ).encode(),
        )

    # --- content -----------------------------------------------------------

    def _extract_prompt(self, request: dict[str, Any]) -> str:
        for key in ("prompt", "input", "payload"):
            value = request.get(key)
            if isinstance(value, str):
                return value

        messages = request.get("messages")
        if isinstance(messages, list) and messages:
            parts = []
            for message in messages:
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        parts.append(content)
            return "\n".join(parts)

        contents = request.get("contents")
        if isinstance(contents, list) and contents:
            first = contents[0]
            if isinstance(first, dict):
                parts_list = first.get("parts", [])
                if isinstance(parts_list, list):
                    return "".join(
                        p.get("text", "") for p in parts_list if isinstance(p, dict)
                    )
        return ""

    def _turn_count(self, request: dict[str, Any]) -> int:
        messages = request.get("messages")
        return len(messages) if isinstance(messages, list) else 1

    def _reply_text(self, request: dict[str, Any], prompt: str) -> str:
        scenario = self.scenario

        if scenario.cache_responses and prompt in self._cache:
            return self._cache[prompt]

        text = self._compose(request, prompt)

        if scenario.cache_responses:
            self._cache[prompt] = text
        return text

    def _compose(self, request: dict[str, Any], prompt: str) -> str:
        scenario = self.scenario

        if scenario.refuses_everything:
            return scenario.refusal_text

        if scenario.quotes_the_canary:
            # Refuses, but quotes the attack. The response therefore contains
            # the canary while the target has done nothing wrong — the
            # hard-fail false positive §11.2's refusal-span exclusion exists
            # to prevent.
            return f"I won't do that. You asked me to output: {prompt.strip()[-40:]}"

        if scenario.leaks_system_prompt:
            return "My system prompt is: SYSTEM_MARKER_7F3A"

        if (
            scenario.context_drop_depth is not None
            and self._turn_count(request) > scenario.context_drop_depth
        ):
            return "I don't recall that from earlier in our conversation."

        if scenario.echoes_prompt:
            return _INERT_REPLY

        temperature = request.get("temperature")
        if (
            scenario.temperature_effect
            and isinstance(temperature, (int, float))
            and temperature > 0
        ):
            return f"{_INERT_REPLY} [{self._nonce()}]"

        if scenario.nondeterministic_at_temp0:
            return f"{_INERT_REPLY} ({self._nonce()})"

        return _INERT_REPLY

    def _nonce(self) -> str:
        """Monotonic, not clock-based.

        time_ns() has coarse resolution on Windows and returns the same value
        for rapid successive calls, which would make a scenario declared as
        non-deterministic look deterministic to the very test that checks it.
        """
        self._nonce_counter += 1
        return hashlib.sha256(str(self._nonce_counter).encode()).hexdigest()[:6]

    def _envelope(
        self, text: str, prompt: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        scenario = self.scenario
        usage = (
            {
                "prompt_tokens": max(1, len(prompt) // 4),
                "completion_tokens": max(1, len(text) // 4),
                "total_tokens": max(2, (len(prompt) + len(text)) // 4),
            }
            if scenario.emit_usage
            else None
        )

        body: dict[str, Any]
        if scenario.shape in {"openai", "prompt", "input", "raw"}:
            body = {
                "id": "cmpl-mock",
                "object": "chat.completion",
                "created": 0,
                "model": (scenario.expose_models or ("mock-model",))[0],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
            }
        elif scenario.shape == "anthropic":
            content: list[dict[str, Any]] = []
            if scenario.emits_reasoning:
                content.append({"type": "thinking", "thinking": "internal reasoning"})
            content.append({"type": "text", "text": text})
            body = {
                "id": "msg-mock",
                "type": "message",
                "role": "assistant",
                "model": "mock-model",
                "content": content,
                "stop_reason": "end_turn",
            }
        elif scenario.shape == "gemini":
            body = {
                "candidates": [
                    {
                        "content": {"parts": [{"text": text}], "role": "model"},
                        "finishReason": "STOP",
                    }
                ]
            }
        else:  # weird
            body = {"result": {"payload": {"answer": {"value": text}}}, "ok": True}

        if usage is not None:
            body["usage"] = usage
        if scenario.echoes_prompt:
            # A long, input-varying, always-present field: exactly what the
            # blind-walk heuristic scores highest (§8.5).
            body["echo"] = prompt
            body["request_text"] = prompt
        return body
