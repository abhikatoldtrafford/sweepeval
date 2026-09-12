"""Authorization gate (spec §18, D39). I8 load-bearing."""

from __future__ import annotations

from pathlib import Path

import pytest

from sweepeval.execute.authz import (
    AuthorizationRequired,
    AuthorizationStore,
    is_local,
    require_authorization,
)


def _never_prompted(_: str) -> str:
    raise AssertionError("should not have prompted")


# --- localhost is exempt ---------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8080/chat",
        "http://127.0.0.1:8080/chat",
        "http://[::1]:8080/chat",
        "http://mock.localhost/chat",
    ],
)
def test_localhost_targets_never_need_affirmation(url: str) -> None:
    record = require_authorization(url, input_fn=_never_prompted)
    assert record.method == "localhost"


def test_a_public_host_is_not_local() -> None:
    assert is_local("https://api.example.com/chat") is False


# --- the gate --------------------------------------------------------------


def test_a_public_host_without_authorization_raises() -> None:
    with pytest.raises(AuthorizationRequired, match=r"api\.example\.com"):
        require_authorization("https://api.example.com/chat", prompt=False)


def test_the_error_names_both_the_interactive_and_the_ci_route() -> None:
    """A gate nobody can satisfy in CI just gets bypassed some other way."""
    with pytest.raises(AuthorizationRequired) as excinfo:
        require_authorization("https://api.example.com/chat", prompt=False)
    message = str(excinfo.value)
    assert "--i-am-authorized" in message
    assert "authorized_hosts" in message
    assert "localhost" in message


def test_the_flag_authorises_without_prompting() -> None:
    record = require_authorization(
        "https://api.example.com/chat", flag=True, input_fn=_never_prompted
    )
    assert record.method == "flag"


def test_a_configured_host_authorises_without_prompting() -> None:
    record = require_authorization(
        "https://api.example.com/chat",
        config_hosts=frozenset({"api.example.com"}),
        input_fn=_never_prompted,
    )
    assert record.method == "config"


def test_an_interactive_yes_authorises() -> None:
    record = require_authorization(
        "https://api.example.com/chat", input_fn=lambda _: "yes"
    )
    assert record.method == "interactive"


def test_anything_other_than_yes_refuses() -> None:
    for answer in ("no", "", "maybe", "sure"):
        with pytest.raises(AuthorizationRequired):
            require_authorization(
                "https://api.example.com/chat", input_fn=lambda _, a=answer: a
            )


def test_the_prompt_says_what_the_suite_actually_does() -> None:
    """An affirmation people cannot read is not informed consent."""
    seen: list[str] = []

    def capture(text: str) -> str:
        seen.append(text)
        return "yes"

    require_authorization("https://api.example.com/chat", input_fn=capture)
    prompt = seen[0]
    assert "prompt-injection" in prompt
    assert "tool calls" in prompt
    assert "api.example.com" in prompt


# --- remembered per host ---------------------------------------------------


def test_an_affirmation_is_remembered_so_it_asks_once(tmp_path: Path) -> None:
    """A prompt on every run becomes one people learn to dismiss."""
    store = AuthorizationStore(tmp_path)
    require_authorization(
        "https://api.example.com/chat", store=store, input_fn=lambda _: "yes"
    )
    again = require_authorization(
        "https://api.example.com/chat", store=store, input_fn=_never_prompted
    )
    assert again.method == "interactive"


def test_authorization_does_not_transfer_between_hosts(tmp_path: Path) -> None:
    store = AuthorizationStore(tmp_path)
    require_authorization(
        "https://api.example.com/chat", store=store, input_fn=lambda _: "yes"
    )
    with pytest.raises(AuthorizationRequired):
        require_authorization(
            "https://other.example.com/chat", store=store, prompt=False
        )


def test_the_flag_is_also_remembered(tmp_path: Path) -> None:
    store = AuthorizationStore(tmp_path)
    require_authorization("https://api.example.com/chat", store=store, flag=True)
    assert store.get("api.example.com") is not None


def test_a_corrupt_store_does_not_grant_authorization(tmp_path: Path) -> None:
    """Failing open on an unreadable file would be the wrong direction."""
    (tmp_path / "authorized_hosts.json").write_text("{ not json", encoding="utf-8")
    store = AuthorizationStore(tmp_path)
    with pytest.raises(AuthorizationRequired):
        require_authorization(
            "https://api.example.com/chat", store=store, prompt=False
        )


def test_the_record_goes_into_the_manifest(tmp_path: Path) -> None:
    record = require_authorization(
        "https://api.example.com/chat", flag=True, store=AuthorizationStore(tmp_path)
    )
    entry = record.to_manifest()
    assert set(entry) == {"host", "affirmed_at", "method"}
    assert entry["affirmed_at"]


def test_no_temp_file_is_left_behind(tmp_path: Path) -> None:
    store = AuthorizationStore(tmp_path)
    require_authorization("https://api.example.com/chat", store=store, flag=True)
    assert not list(tmp_path.glob("*.tmp"))
