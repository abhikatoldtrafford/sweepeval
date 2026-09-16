"""Reading back the sources a target cited (§11, family 6).

Retrieval differs from every other family here in one way that decides the
whole design: **sweepeval cannot supply the corpus.** For tool integrity it
declares the tool schemas, which is what makes "did this call conform"
answerable. There is no equivalent for retrieval -- the documents belong to the
target's own pipeline, and a black-box client has no way to inject them or to
know what should have been retrieved.

So precision@k, recall@k, MRR and nDCG are **not measurable here**, and the
scorer says so rather than approximating them. Every one of those needs
relevance labels over the target's corpus. A number computed without them
would be a number about a corpus we invented.

What *is* measurable from outside, and what this module reads:

  is anything cited at all, and through which channel
  are the citations structurally sound -- a source that identifies something,
      spans that land inside the answer they annotate
  does the same question produce the same sources twice
  does the target fabricate a citation for a question nothing could source

Five channels, because "has retrieval" is not one wire format:

  ``openai.annotations``   ``message.annotations[].url_citation``, which is
                           what a live search-backed model returns and what the
                           old detector missed entirely -- it looked for
                           ``documents``/``sources``/``citations`` and a real
                           retrieval endpoint has none of those keys
  ``anthropic.citations``  ``citations`` inside a content block
  ``gemini.grounding``     ``groundingMetadata.groundingChunks[].web``
  ``generic.documents``    a top-level array of documents, the shape most
                           self-hosted RAG services return
  ``text.inline``          no structured field at all: markdown links or
                           ``[1]``-style markers in the prose

The last one is the honesty case, exactly as with tool calling. A target that
writes references into its answer has not surfaced retrievable sources, and
crediting it would report a capability it does not have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "IR_METRICS_REASON",
    "Citation",
    "extract_citations",
    "validate_citation",
]

IR_METRICS_REASON = (
    "precision@k, recall@k, MRR and nDCG need relevance labels over the "
    "target's own corpus; a black-box client cannot supply the corpus or know "
    "what should have been retrieved, so they are not computed rather than "
    "approximated"
)

DOCUMENT_KEYS = (
    "documents", "sources", "retrieved", "chunks", "search_results",
    "references", "contexts", "passages",
)
"""Array keys a self-hosted retrieval service is likely to return.

Deliberately broad. The cost of a name this list misses is a false
UNSUPPORTED, which is the defect this module exists to fix; the cost of an
extra name is a citation found in something that was not one, which
:func:`validate_citation` then has to reject on its own merits.
"""


@dataclass(frozen=True)
class Citation:
    """One source a target pointed at, normalised across channels."""

    source: str
    """Whatever identifies it: a URL, a document id, a title. Never empty for
    a citation that validates."""

    channel: str
    title: str = ""
    start: int | None = None
    end: int | None = None
    """Character span in the answer this citation annotates, when declared."""

    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.problems


def extract_citations(payload: Any, text: str = "") -> list[Citation]:
    """Every citation in a response, in whichever channel carried it.

    Structured channels first and prose last, for the same reason tool calls
    are read that way: a target doing both should be credited with the real
    one.
    """
    for reader in (
        _openai_annotations,
        _anthropic_citations,
        _gemini_grounding,
        _generic_documents,
    ):
        found = reader(payload)
        if found:
            return found
    return _inline_in_text(text or _message_text(payload))


def _openai_annotations(payload: Any) -> list[Citation]:
    out: list[Citation] = []
    for choice in _get(payload, "choices") or []:
        message = _get(choice, "message") or {}
        for entry in _get(message, "annotations") or []:
            if not isinstance(entry, dict):
                continue
            body = entry.get("url_citation") or entry.get("file_citation") or {}
            if not isinstance(body, dict):
                continue
            out.append(
                Citation(
                    source=str(body.get("url") or body.get("file_id") or ""),
                    title=str(body.get("title") or ""),
                    start=_index(body.get("start_index")),
                    end=_index(body.get("end_index")),
                    channel="openai.annotations",
                )
            )
    return out


def _anthropic_citations(payload: Any) -> list[Citation]:
    out: list[Citation] = []
    for block in _get(payload, "content") or []:
        if not isinstance(block, dict):
            continue
        for entry in block.get("citations") or []:
            if not isinstance(entry, dict):
                continue
            out.append(
                Citation(
                    source=str(
                        entry.get("url")
                        or entry.get("document_title")
                        or entry.get("document_index", "")
                    ),
                    title=str(entry.get("document_title") or entry.get("title") or ""),
                    start=_index(entry.get("start_char_index")),
                    end=_index(entry.get("end_char_index")),
                    channel="anthropic.citations",
                )
            )
    return out


def _gemini_grounding(payload: Any) -> list[Citation]:
    out: list[Citation] = []
    for candidate in _get(payload, "candidates") or []:
        metadata = _get(candidate, "groundingMetadata") or {}
        for chunk in _get(metadata, "groundingChunks") or []:
            web = _get(chunk, "web") or {}
            if web:
                out.append(
                    Citation(
                        source=str(web.get("uri") or ""),
                        title=str(web.get("title") or ""),
                        channel="gemini.grounding",
                    )
                )
    return out


def _generic_documents(payload: Any) -> list[Citation]:
    """A documents array anywhere in the response.

    Walked rather than read at a fixed path: self-hosted services put it at
    the top level, under `data`, or beside the message, and which one is not
    knowable from outside.
    """
    out: list[Citation] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in DOCUMENT_KEYS and isinstance(value, list):
                    out.extend(_documents(value, key))
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return out


def _documents(entries: list[Any], key: str) -> list[Citation]:
    out: list[Citation] = []
    for entry in entries:
        if isinstance(entry, str):
            out.append(Citation(source=entry, channel=f"generic.{key}"))
        elif isinstance(entry, dict):
            out.append(
                Citation(
                    source=str(
                        entry.get("url")
                        or entry.get("id")
                        or entry.get("doc_id")
                        or entry.get("title")
                        or ""
                    ),
                    title=str(entry.get("title") or ""),
                    channel=f"generic.{key}",
                )
            )
    return out


_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\((https?://[^\s)]+)\)")
_BARE_URL = re.compile(r"(?<![(\w])(https?://[^\s<>\]),]+)")


def _inline_in_text(text: str) -> list[Citation]:
    """References a target wrote into its answer rather than surfacing.

    Not retrieval, and not nothing -- the same distinction tool calling draws
    between emitting a call and describing one. Recorded so the report can say
    which it was.
    """
    if not text:
        return []
    urls: list[str] = []
    for match in _MARKDOWN_LINK.finditer(text):
        urls.append(match.group(1))
    for match in _BARE_URL.finditer(text):
        if match.group(1) not in urls:
            urls.append(match.group(1))
    return [
        Citation(source=url, channel="text.inline")
        for url in dict.fromkeys(urls)
    ]


def validate_citation(citation: Citation, answer: str) -> Citation:
    """Check what can be checked without leaving the target.

    **Nothing here fetches the cited source.** sweepeval talks to the endpoint
    you named and to nothing else; resolving a citation would mean issuing
    requests to third parties on the user's behalf, from a tool whose whole
    claim is that it makes no other network calls. So "this URL exists" and
    "this document says what the answer claims" are out of scope, and the
    report says so rather than implying the citation was verified.

    What is left is still worth having: a citation that identifies nothing, or
    whose span does not land in the answer it annotates, is malformed on its
    own terms.
    """
    problems: list[str] = []

    if not citation.source.strip():
        problems.append("the citation identifies no source")

    if citation.start is not None or citation.end is not None:
        start, end = citation.start, citation.end
        if start is None or end is None:
            problems.append("the citation declares half a span")
        elif start < 0 or end < 0:
            problems.append(f"the span ({start}, {end}) is negative")
        elif start >= end:
            problems.append(f"the span ({start}, {end}) does not advance")
        elif end > len(answer):
            problems.append(
                f"the span ends at {end}, past the {len(answer)}-character answer"
            )

    return Citation(
        source=citation.source, channel=citation.channel, title=citation.title,
        start=citation.start, end=citation.end, problems=tuple(problems),
    )


def _index(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _get(node: Any, key: str) -> Any:
    return node.get(key) if isinstance(node, dict) else None


def _message_text(payload: Any) -> str:
    for choice in _get(payload, "choices") or []:
        message = _get(choice, "message") or {}
        content = _get(message, "content")
        if isinstance(content, str) and content:
            return content
    return ""
