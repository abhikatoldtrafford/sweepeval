"""Authorization gate (spec §18, D39). I8 load-bearing.

The security family is an active prompt-injection and exfiltration suite, and
it gets pointed at whatever URL the user types. On an agent target even an
inert discovery prompt can trigger tool calls, writes, or spend on the target's
side.

So: before the security family runs against a **non-localhost** host, the user
affirms authorization. Recorded in the manifest with a timestamp and remembered
per host, so it asks once rather than becoming a prompt people learn to dismiss.

This is an adoption prerequisite as much as an ethical one. Enterprises will not
run an unlabelled attack tool.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

__all__ = [
    "LOCAL_HOSTS",
    "AuthorizationRecord",
    "AuthorizationRequired",
    "AuthorizationStore",
    "is_local",
    "require_authorization",
]

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"})

_AFFIRMATION = """\
sweepeval's security suite sends live prompt-injection and exfiltration probes
to {host}.

On an agent system those requests may trigger real tool calls, writes, or spend
on the target's side. Only run this against systems you are authorised to test.

Type 'yes' to confirm you are authorised: """


class AuthorizationRequired(RuntimeError):
    """Raised when the security family would run without authorization."""

    def __init__(self, host: str) -> None:
        self.host = host
        super().__init__(
            f"the security family needs authorization for {host}.\n"
            "  interactive: re-run and answer the prompt\n"
            "  CI: pass --i-am-authorized, or set authorized_hosts in the config\n"
            f"  (localhost targets never need this)"
        )


@dataclass(frozen=True)
class AuthorizationRecord:
    host: str
    affirmed_at: str
    method: str
    """``interactive``, ``flag``, ``config`` or ``localhost``."""

    def to_manifest(self) -> dict[str, str]:
        return {
            "host": self.host,
            "affirmed_at": self.affirmed_at,
            "method": self.method,
        }


def is_local(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return host in LOCAL_HOSTS or host.endswith(".localhost")


class AuthorizationStore:
    """Per-host affirmations, remembered between runs."""

    def __init__(self, root: Path) -> None:
        self._path = Path(root) / "authorized_hosts.json"

    def _load(self) -> dict[str, dict[str, str]]:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def get(self, host: str) -> AuthorizationRecord | None:
        entry = self._load().get(host)
        if not entry:
            return None
        return AuthorizationRecord(
            host=host,
            affirmed_at=entry.get("affirmed_at", ""),
            method=entry.get("method", "unknown"),
        )

    def remember(self, record: AuthorizationRecord) -> None:
        data = self._load()
        data[record.host] = {
            "affirmed_at": record.affirmed_at,
            "method": record.method,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._path)


def require_authorization(
    url: str,
    *,
    store: AuthorizationStore | None = None,
    flag: bool = False,
    config_hosts: frozenset[str] = frozenset(),
    prompt: bool = True,
    input_fn: object = None,
) -> AuthorizationRecord:
    """Return the record, or raise :class:`AuthorizationRequired`.

    Order: localhost is exempt, then ``--i-am-authorized``, then the config's
    declared hosts, then a remembered affirmation, then an interactive prompt.
    """
    host = (urlsplit(url).hostname or url).lower()
    now = datetime.now(timezone.utc).isoformat()

    if is_local(url):
        return AuthorizationRecord(host, now, "localhost")

    if flag:
        record = AuthorizationRecord(host, now, "flag")
        if store:
            store.remember(record)
        return record

    if host in config_hosts:
        return AuthorizationRecord(host, now, "config")

    if store:
        remembered = store.get(host)
        if remembered:
            return remembered

    if not prompt:
        raise AuthorizationRequired(host)

    reader = input_fn if callable(input_fn) else input
    answer = str(reader(_AFFIRMATION.format(host=host))).strip().lower()
    if answer not in {"yes", "y"}:
        raise AuthorizationRequired(host)

    record = AuthorizationRecord(host, now, "interactive")
    if store:
        store.remember(record)
    return record
