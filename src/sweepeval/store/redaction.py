"""Secret redaction (spec §6.6, §8.4, D38).

Every artifact writer goes through this. Two things make it necessary rather
than defensive: ``--key`` may end up in a query parameter, because §8.4's auth
ladder tries a query param as its fourth rung; and ``baseline.json`` is
designed to be committed to the user's repository.

Built before the stores on purpose. If the stores land first, some writer
bypasses the redactor and the leak ships — so ``BlobStore`` and the JSONL log
both *require* a ``Redactor`` in their constructor.

**Judgement call, flagged:** the spec does not set a minimum length for
substring masking of a supplied secret. This uses 8. Shorter and a
three-character "key" would mask half the probe corpus; longer and genuinely
short keys leak.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sweepeval.schema.hashing import sha256_hex

__all__ = ["AUTH_PARAM_NAMES", "MASK", "MIN_SECRET_LENGTH", "Redactor"]

MASK = "***"
MIN_SECRET_LENGTH = 8

AUTH_PARAM_NAMES = frozenset(
    {
        "api_key",
        "api-key",
        "apikey",
        "key",
        "access_token",
        "token",
        "auth",
        "authorization",
        "x-api-key",
        "subscription-key",
    }
)

# Provider-shaped tokens. Masked even when not supplied as --key, because a
# target's own error body can echo a credential back at us.
_TOKEN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(bearer\s+)\S{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bxoxb-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
)


class Redactor:
    """Masks secrets in URLs, free text and nested mappings."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets = tuple(
            sorted(
                {s for s in secrets if s and len(s) >= MIN_SECRET_LENGTH},
                key=len,
                reverse=True,
            )
        )

    # --- URLs -------------------------------------------------------------

    def url(self, url: str) -> str:
        """Strip userinfo credentials and mask auth query parameters."""
        parts = urlsplit(url)

        netloc = parts.netloc
        if "@" in netloc:
            netloc = netloc.rsplit("@", 1)[1]

        # safe="*" keeps the mask readable: urlencode would otherwise write
        # %2A%2A%2A, and a redacted URL is shown in manifests and reports.
        query = urlencode(
            [
                (k, MASK if self._is_secret_param(k, v) else v)
                for k, v in parse_qsl(parts.query, keep_blank_values=True)
            ],
            safe="*",
        )

        return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))

    def _is_secret_param(self, name: str, value: str) -> bool:
        return name.lower() in AUTH_PARAM_NAMES or value in self._secrets

    def endpoint_fingerprint(self, url: str) -> str:
        """Credential-free identity for a target (§6.4).

        Query and userinfo are dropped entirely rather than masked, so the same
        endpoint reached with different keys fingerprints identically.
        """
        parts = urlsplit(url)
        host = parts.netloc.rsplit("@", 1)[-1]
        return sha256_hex(f"{parts.scheme}://{host}{parts.path}".encode())

    # --- text and structures ---------------------------------------------

    def text(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, MASK)
        for pattern in _TOKEN_PATTERNS:
            text = pattern.sub(
                lambda m: (m.group(1) + MASK) if m.groups() else MASK, text
            )
        return text

    def mapping(self, data: Mapping[str, Any]) -> dict[str, Any]:
        return {k: self._value(k, v) for k, v in data.items()}

    def _value(self, key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            return self.mapping(value)
        if isinstance(value, (list, tuple)):
            return type(value)(self._value(key, v) for v in value)
        if isinstance(value, str):
            if key.lower() in AUTH_PARAM_NAMES or value in self._secrets:
                return MASK
            return self.text(value)
        return value
