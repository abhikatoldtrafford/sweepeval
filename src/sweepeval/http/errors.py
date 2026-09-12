"""Error classification (spec §7).

One table, consulted everywhere. Two entries carry weight beyond their
obviousness:

* ``refusal`` is **not** an error. A target that declines a prompt-injection
  attempt is behaving correctly, and scoring that as a failure would invert the
  security metric.
* ``malformed`` is its own class, so a 200 whose body fails extraction is never
  quietly scored as a zero.
"""

from __future__ import annotations

import httpx

from sweepeval.schema.call import ErrorClass

__all__ = ["RETRYABLE_STATUSES", "TERMINAL_STATUSES", "classify_exception", "classify_status"]

RETRYABLE_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
TERMINAL_STATUSES = frozenset({400, 401, 403, 404, 405, 406, 410, 413, 415, 422, 501})


def classify_status(status: int) -> ErrorClass:
    """Map an HTTP status to its class.

    Unknown 4xx defaults to ``terminal`` and unknown 5xx to ``retryable``:
    retrying a request the server has already rejected on its merits wastes
    budget, while a server-side fault is usually transient.
    """
    if 200 <= status < 300:
        return ErrorClass.ok
    if status in RETRYABLE_STATUSES:
        return ErrorClass.retryable
    if status in TERMINAL_STATUSES:
        return ErrorClass.terminal
    if 500 <= status < 600:
        return ErrorClass.retryable
    if 400 <= status < 500:
        return ErrorClass.terminal
    # 1xx and 3xx should not reach here — httpx follows redirects — but a
    # surprising status is safer treated as terminal than retried in a loop.
    return ErrorClass.terminal


def classify_exception(exc: BaseException) -> ErrorClass:
    """Map a transport exception to its class."""
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)):
        return ErrorClass.retryable
    if isinstance(exc, httpx.HTTPError):
        return ErrorClass.terminal
    return ErrorClass.terminal
