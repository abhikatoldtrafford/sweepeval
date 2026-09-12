"""Shared pytest configuration and the in-process mock harness (spec §17).

Every network-touching test mounts the mock through ``httpx.ASGITransport``,
so the suite runs with zero sockets and zero tokens.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from sweepeval.mock.app import MockApp
from sweepeval.mock.scenario import Scenario, load_scenario, load_scenario_dir

SCENARIO_DIR = Path(__file__).parent / "scenarios"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--snapshot-update",
        action="store_true",
        default=False,
        help="Rewrite golden snapshots instead of asserting against them.",
    )


@pytest.fixture(scope="session")
def scenarios() -> dict[str, Scenario]:
    return load_scenario_dir(SCENARIO_DIR)


@pytest.fixture
def scenario_path() -> Path:
    return SCENARIO_DIR


def make_app(name: str) -> MockApp:
    """Build a fresh mock for one scenario. Fresh, because the app is stateful."""
    return MockApp(load_scenario(SCENARIO_DIR / f"{name}.yaml"))


def make_client(app: MockApp, base_url: str = "https://mock.test") -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url=base_url
    )


@pytest.fixture
def mock_factory() -> Iterator[object]:
    """Factory yielding ``(app, client)`` for a named scenario."""
    clients: list[httpx.AsyncClient] = []

    def build(name: str) -> tuple[MockApp, httpx.AsyncClient]:
        app = make_app(name)
        client = make_client(app)
        clients.append(client)
        return app, client

    yield build

    # Clients are closed by the tests that own them; this is a belt-and-braces
    # sweep for any that raised before their own aclose().
    for client in clients:
        if not client.is_closed:
            with contextlib.suppress(RuntimeError):
                asyncio.get_event_loop().run_until_complete(client.aclose())
