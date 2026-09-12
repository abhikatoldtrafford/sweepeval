"""Auth probing (spec §8.4).

Bearer → ``x-api-key`` → ``api-key`` → query parameter, first success wins.

The fourth rung matters beyond its rank: a query-param win puts the key in the
URL, and ``baseline.json`` is designed to be committed. So the winner is
recorded and, when it is the query param, the redactor is activated everywhere
downstream (§6.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["AUTH_LADDER", "AuthMethod", "apply_auth", "auth_ladder"]


@dataclass(frozen=True)
class AuthMethod:
    name: str
    puts_key_in_url: bool = False

    def apply(self, key: str) -> tuple[dict[str, str], dict[str, Any]]:
        """Return ``(headers, params)`` carrying the key."""
        if self.name == "bearer":
            return {"authorization": f"Bearer {key}"}, {}
        if self.name == "x-api-key":
            return {"x-api-key": key}, {}
        if self.name == "api-key":
            return {"api-key": key}, {}
        if self.name == "query":
            return {}, {"api_key": key}
        if self.name == "none":
            return {}, {}
        raise ValueError(f"unknown auth method {self.name!r}")


AUTH_LADDER: tuple[AuthMethod, ...] = (
    AuthMethod("bearer"),
    AuthMethod("x-api-key"),
    AuthMethod("api-key"),
    AuthMethod("query", puts_key_in_url=True),
)


def auth_ladder(key: str | None) -> tuple[AuthMethod, ...]:
    """The ladder to try. With no key, only the unauthenticated attempt."""
    if not key:
        return (AuthMethod("none"),)
    return AUTH_LADDER


def apply_auth(
    method: AuthMethod, key: str | None
) -> tuple[dict[str, str], dict[str, Any]]:
    if not key or method.name == "none":
        return {}, {}
    return method.apply(key)
