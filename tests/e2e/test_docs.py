"""The documentation has to be executable and internally consistent (§19.2).

Two failure modes this catches, both of which are invisible to a reader until
they try to use the docs:

* a cookbook snippet that no longer matches the protocol it claims to
  implement — the most common way a plugin guide rots;
* a nav entry or an internal link pointing at a page that does not exist.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _python_blocks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


# --- the plugin cookbook runs (§19.2, I10) --------------------------------


def test_the_cookbook_snippets_execute() -> None:
    """A guide nobody has run is a guide that does not work."""
    blocks = _python_blocks(DOCS / "guides" / "plugins.md")
    assert len(blocks) >= 3, "the cookbook lost its snippets"

    namespace: dict[str, object] = {}
    for index, block in enumerate(blocks):
        if "REGISTRY.register" in block or "from sweepeval.scorers import register" in block:
            # Registration mutates a process-global registry; the objects are
            # what this test is about, not the side effect.
            continue
        exec(compile(block, f"plugins.md#{index}", "exec"), namespace)

    assert "BrandVoiceScorer" in namespace
    assert "AcmeCompletion" in namespace


def test_the_cookbook_scorer_satisfies_the_protocol() -> None:
    from sweepeval.scorers.base import Scorer

    namespace: dict[str, object] = {}
    for block in _python_blocks(DOCS / "guides" / "plugins.md"):
        if "class BrandVoiceScorer" in block:
            exec(compile(block, "plugins.md", "exec"), namespace)
    scorer = namespace["BrandVoiceScorer"]()  # type: ignore[operator]
    assert isinstance(scorer, Scorer)
    assert scorer.metrics()


def test_the_cookbook_scorer_actually_scores() -> None:
    """It has to discriminate, not merely typecheck."""
    from sweepeval.schema.observation import Verdict
    from sweepeval.scorers.base import ScoreContext

    namespace: dict[str, object] = {}
    for block in _python_blocks(DOCS / "guides" / "plugins.md"):
        if "class BrandVoiceScorer" in block:
            exec(compile(block, "plugins.md", "exec"), namespace)
    scorer = namespace["BrandVoiceScorer"]()  # type: ignore[operator]

    def score(text: str) -> Verdict:
        context = ScoreContext(
            run_id="r", config_id="c", run_idx=0, text=text, ts="2026-09-12T00:00:00Z"
        )
        return scorer.score(None, [], context)[0].verdict

    assert score("Happy to help with that.") is Verdict.PASS
    assert score("Let me circle back on the synergy here.") is Verdict.FAIL
    assert score("   ") is Verdict.UNSCORABLE


def test_the_cookbook_shape_satisfies_the_protocol() -> None:
    from sweepeval.discovery.shapes.base import Shape

    namespace: dict[str, object] = {}
    for block in _python_blocks(DOCS / "guides" / "plugins.md"):
        if "class AcmeCompletion" in block:
            exec(compile(block, "plugins.md", "exec"), namespace)
    shape = namespace["AcmeCompletion"]()  # type: ignore[operator]

    assert isinstance(shape, Shape)
    body = shape.build("hello", temperature=0.7)
    assert body["query"] == "hello"
    assert body["opts"]["temperature"] == 0.7
    multi = shape.build_multi_turn([("user", "a"), ("assistant", "b")])
    assert "a" in multi["query"] and "b" in multi["query"]
    assert shape.prior_text_paths()


def test_the_cookbook_snippets_stay_short() -> None:
    """Under 50 lines each, or they stop being a cookbook."""
    for block in _python_blocks(DOCS / "guides" / "plugins.md"):
        assert len(block.splitlines()) <= 60, block[:120]


# --- the nav and the links resolve ----------------------------------------


def _nav_pages(node: object) -> list[str]:
    out: list[str] = []
    if isinstance(node, str):
        out.append(node)
    elif isinstance(node, list):
        for item in node:
            out.extend(_nav_pages(item))
    elif isinstance(node, dict):
        for value in node.values():
            out.extend(_nav_pages(value))
    return out


def test_every_nav_entry_exists() -> None:
    """A docs site that 404s on its own navigation is worse than none."""
    config = yaml.safe_load(
        (ROOT / "mkdocs.yml").read_text(encoding="utf-8").replace("!!python/name:", "")
    )
    missing = [page for page in _nav_pages(config["nav"]) if not (DOCS / page).exists()]
    assert not missing, missing


@pytest.mark.parametrize(
    "page", sorted(p.relative_to(DOCS).as_posix() for p in DOCS.rglob("*.md"))
)
def test_internal_links_resolve(page: str) -> None:
    source = DOCS / page
    text = source.read_text(encoding="utf-8")
    broken = []
    for target in re.findall(r"\]\(([^)#]+\.md)(?:#[^)]*)?\)", text):
        if target.startswith(("http://", "https://")):
            continue
        if not (source.parent / target).resolve().exists():
            broken.append(target)
    assert not broken, f"{page}: {broken}"


def test_the_readme_links_resolve() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    broken = [
        target
        for target in re.findall(r"\]\(([^)#]+)\)", text)
        if not target.startswith(("http://", "https://", "mailto:"))
        and not (ROOT / target).exists()
    ]
    assert not broken, broken
