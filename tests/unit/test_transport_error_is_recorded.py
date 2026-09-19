"""A transport failure must say what it was, not only that it happened.

`error_excerpt` is the head of a failed response body, and a transport failure
has no body -- so the excerpt was None and `classify_exception` reduced the
exception to one of a handful of classes. The run then recorded "retryable"
and nothing else: a pool timeout, a dropped connection and a DNS failure are
indistinguishable in the artifact, in a tool whose whole promise is diagnosing
a target from the stored run.

Found in a live four-model sweep that logged nine of them -- the first run of
this project to see any -- and could say nothing about them afterwards.
"""

from __future__ import annotations

import httpx
import pytest

from sweepeval.http.client import TransportClient
from sweepeval.http.governor import Governor
from sweepeval.schema.call import ErrorClass

URL = "https://mock.test/v1/chat/completions"


class _Boom(httpx.AsyncBaseTransport):
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise self.exc


async def _send(transport: httpx.AsyncBaseTransport):
    client = httpx.AsyncClient(transport=transport, base_url="https://mock.test")
    governor = Governor()
    governor.backoff_base_s = 0.0  # the policy, not the sleeping, is the subject
    try:
        return await TransportClient(client, governor, run_id="r").call(
            URL, {"messages": []}, config_id="cfg-00", unit_id="u",
        )
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("[Errno 11001] getaddrinfo failed"),
        httpx.PoolTimeout("pool timeout"),
        httpx.RemoteProtocolError("server disconnected without sending a response"),
    ],
)
async def test_the_exception_reaches_the_stored_call(exc) -> None:
    results = await _send(_Boom(exc))
    call = results[0].call

    assert call.response.status == 0
    assert call.response.error_excerpt, "a transport failure recorded no cause"
    assert type(exc).__name__ in call.response.error_excerpt


async def test_two_different_failures_are_distinguishable() -> None:
    """The point: not that something is recorded, but that the record tells
    them apart. Both reduce to the same ErrorClass."""
    a = (await _send(_Boom(httpx.PoolTimeout("pool timeout"))))[0].call
    b = (await _send(_Boom(httpx.ConnectError("getaddrinfo failed"))))[0].call

    assert a.response.error_class == b.response.error_class
    assert a.response.error_excerpt != b.response.error_excerpt


async def test_a_successful_call_still_records_no_excerpt() -> None:
    class _Fine(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": "hi"}}]}
            )

    call = (await _send(_Fine()))[0].call
    assert call.response.status == 200
    assert call.response.error_class is ErrorClass.ok
    assert call.response.error_excerpt is None
