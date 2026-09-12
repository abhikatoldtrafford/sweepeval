"""Discovery budget and abort diagnostic (spec §8.3). I8 load-bearing.

I8: "Discovery and capability detection send only inert, read-shaped requests,
never exceed their budget, and never run the security family without
authorization."

Two halves live here. The **inert prompt** is a module constant and the only
text discovery ever sends — a test greps every discovery request body for it,
because on an agent target a read-shaped body is still a live prompt that can
trigger tool calls, writes, or spend.

The **abort diagnostic** is the other half. When the budget is exhausted the
tool must say exactly what it tried: every shape, every mutation, every
response with status, content type and error body. An abort that just says
"could not determine shape" leaves the user with no next move, on the one
command the product is named for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_MAX_POSTS",
    "DEFAULT_TOKEN_CAP",
    "DEFAULT_WALL_CLOCK_S",
    "INERT_PROMPT",
    "Attempt",
    "DiscoveryAborted",
    "DiscoveryBudget",
]

INERT_PROMPT = "Reply with the single word OK."
"""The only prompt discovery sends (§8.3).

Fixed and inert on purpose. Nothing in discovery asks the target to act.
"""

DEFAULT_MAX_POSTS = 25
DEFAULT_WALL_CLOCK_S = 120.0
DEFAULT_TOKEN_CAP = 20_000


@dataclass(frozen=True)
class Attempt:
    """One recorded discovery request, for the diagnostic."""

    stage: str
    shape: str
    path: str
    mutation: str | None
    status: int
    content_type: str
    error_excerpt: str
    note: str = ""

    def render(self) -> str:
        mutation = f" +{self.mutation}" if self.mutation else ""
        excerpt = f" — {self.error_excerpt}" if self.error_excerpt else ""
        return (
            f"  [{self.stage}] {self.shape}{mutation} POST {self.path} "
            f"-> {self.status} {self.content_type}{excerpt}"
        )


class DiscoveryAborted(RuntimeError):
    """Budget exhausted without identifying a shape.

    Carries the full transcript, because the message *is* the deliverable when
    discovery fails.
    """

    def __init__(self, reason: str, attempts: list[Attempt], url: str) -> None:
        self.reason = reason
        self.attempts = attempts
        self.url = url
        super().__init__(self.render())

    def render(self) -> str:
        lines = [
            f"discovery failed against {self.url}: {self.reason}",
            "",
            f"tried {len(self.attempts)} request(s):",
        ]
        lines.extend(attempt.render() for attempt in self.attempts)
        lines += [
            "",
            "next steps:",
            "  - if this is a base URL, give the full endpoint path",
            "  - if the shape is unusual, write sweepeval.yaml by hand "
            "(see docs: custom shapes)",
            "  - if authentication failed, check --key and the auth scheme",
        ]
        return "\n".join(lines)


@dataclass
class DiscoveryBudget:
    """Hard caps on discovery. Never an indefinite loop against a stranger."""

    max_posts: int = DEFAULT_MAX_POSTS
    wall_clock_s: float = DEFAULT_WALL_CLOCK_S
    token_cap: int = DEFAULT_TOKEN_CAP

    posts: int = 0
    tokens: int = 0
    started: float = field(default_factory=time.monotonic)
    attempts: list[Attempt] = field(default_factory=list)

    def remaining_posts(self) -> int:
        return max(0, self.max_posts - self.posts)

    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    def exhausted(self) -> str | None:
        """Return the reason the budget is spent, or None."""
        if self.posts >= self.max_posts:
            return f"request budget exhausted ({self.max_posts} POSTs)"
        if self.elapsed_s() >= self.wall_clock_s:
            return f"wall-clock budget exhausted ({self.wall_clock_s:.0f}s)"
        if self.tokens >= self.token_cap:
            return f"token budget exhausted ({self.token_cap} tokens)"
        return None

    def record(self, attempt: Attempt, *, tokens: int = 0) -> None:
        self.posts += 1
        self.tokens += tokens
        self.attempts.append(attempt)

    def abort(self, url: str, reason: str | None = None) -> DiscoveryAborted:
        return DiscoveryAborted(
            reason or self.exhausted() or "no shape matched",
            list(self.attempts),
            url,
        )
