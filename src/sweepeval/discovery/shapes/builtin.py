"""The six built-in shapes, in spec §8.1's prior-likelihood order.

Priorities are spaced by 10 so a third-party shape can slot between two
built-ins without renumbering them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sweepeval.discovery.shapes.base import register

__all__ = [
    "AnthropicMessages",
    "GeminiGenerateContent",
    "OpenAIChatCompletions",
    "RawText",
    "SimpleInput",
    "SimplePrompt",
]

_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "seed", "model")


def _sampling(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k in _SAMPLING_KEYS and v is not None}


@dataclass
class OpenAIChatCompletions:
    name: str = "openai.chat_completions"
    priority: float = 10.0
    default_paths: tuple[str, ...] = ("/v1/chat/completions", "/chat/completions")

    def build(self, prompt: str, **params: Any) -> dict[str, Any]:
        return self.build_multi_turn([("user", prompt)], **params)

    def build_multi_turn(
        self, turns: list[tuple[str, str]], **params: Any
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": [{"role": role, "content": text} for role, text in turns]
        }
        body.update(_sampling(params))
        return body

    def prior_text_paths(self) -> tuple[str, ...]:
        return ("$.choices[0].message.content", "$.choices[0].text")


@dataclass
class AnthropicMessages:
    name: str = "anthropic.messages"
    priority: float = 20.0
    default_paths: tuple[str, ...] = ("/v1/messages",)

    def build(self, prompt: str, **params: Any) -> dict[str, Any]:
        return self.build_multi_turn([("user", prompt)], **params)

    def build_multi_turn(
        self, turns: list[tuple[str, str]], **params: Any
    ) -> dict[str, Any]:
        system = [t for r, t in turns if r == "system"]
        body: dict[str, Any] = {
            "messages": [
                {"role": role, "content": text} for role, text in turns if role != "system"
            ],
            # Anthropic requires max_tokens; omitting it is a guaranteed 400
            # that would look like a shape mismatch rather than a missing field.
            "max_tokens": params.get("max_tokens", 1024),
        }
        if system:
            body["system"] = "\n".join(system)
        body.update(_sampling(params))
        return body

    def prior_text_paths(self) -> tuple[str, ...]:
        return ("$.content[*].text", "$.content[0].text")


@dataclass
class GeminiGenerateContent:
    name: str = "gemini.generate_content"
    priority: float = 30.0
    default_paths: tuple[str, ...] = ("/v1beta/models/{model}:generateContent",)

    def build(self, prompt: str, **params: Any) -> dict[str, Any]:
        return self.build_multi_turn([("user", prompt)], **params)

    def build_multi_turn(
        self, turns: list[tuple[str, str]], **params: Any
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "contents": [
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": text}],
                }
                for role, text in turns
                if role != "system"
            ]
        }
        # `model` is a sampling key for the shapes that carry it in the body.
        # Gemini names the model in the URL, so passing it here put a `model`
        # field inside `generationConfig` — somewhere the API does not read it
        # and rejects it — which is what a swept or pinned model did to every
        # request against a Gemini target.
        config = {k: v for k, v in _sampling(params).items() if k != "model"}
        if config:
            body["generationConfig"] = config
        return body

    def prior_text_paths(self) -> tuple[str, ...]:
        return ("$.candidates[0].content.parts[0].text",)


@dataclass
class SimplePrompt:
    name: str = "simple.prompt"
    priority: float = 40.0
    default_paths: tuple[str, ...] = ("/generate", "/completions", "/v1/completions")

    def build(self, prompt: str, **params: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"prompt": prompt}
        body.update(_sampling(params))
        return body

    def build_multi_turn(
        self, turns: list[tuple[str, str]], **params: Any
    ) -> dict[str, Any]:
        # No history slot: flatten. Discovery records this so the multi-turn
        # detector knows replay is unavailable for this shape (§9.2).
        joined = "\n".join(f"{role}: {text}" for role, text in turns)
        return self.build(joined, **params)

    def prior_text_paths(self) -> tuple[str, ...]:
        return ("$.text", "$.completion", "$.output", "$.response")


@dataclass
class SimpleInput:
    name: str = "simple.input"
    priority: float = 50.0
    default_paths: tuple[str, ...] = ("/invoke", "/predict", "/run")

    def build(self, prompt: str, **params: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"input": prompt}
        body.update(_sampling(params))
        return body

    def build_multi_turn(
        self, turns: list[tuple[str, str]], **params: Any
    ) -> dict[str, Any]:
        joined = "\n".join(f"{role}: {text}" for role, text in turns)
        return self.build(joined, **params)

    def prior_text_paths(self) -> tuple[str, ...]:
        return ("$.output", "$.output_text", "$.result", "$.response")


@dataclass
class RawText:
    name: str = "raw.text"
    priority: float = 60.0
    default_paths: tuple[str, ...] = ("/",)
    _marker: bool = field(default=True, repr=False)

    def build(self, prompt: str, **params: Any) -> dict[str, Any]:
        # The client posts JSON; a raw-text target is represented as a body
        # with a single well-known key that the client unwraps.
        return {"__raw__": prompt}

    def build_multi_turn(
        self, turns: list[tuple[str, str]], **params: Any
    ) -> dict[str, Any]:
        return self.build("\n".join(text for _, text in turns), **params)

    def prior_text_paths(self) -> tuple[str, ...]:
        return ("$",)


for _shape in (
    OpenAIChatCompletions(),
    AnthropicMessages(),
    GeminiGenerateContent(),
    SimplePrompt(),
    SimpleInput(),
    RawText(),
):
    register(_shape)

del _shape
