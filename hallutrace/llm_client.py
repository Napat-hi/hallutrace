"""One client for every LLM call HALoGEN's verifiers make.

HALoGEN's verifiers build `together.Together(...)` themselves and call
`completions.create` or `chat.completions.create`. LLMClient has the same two
methods; route() puts it where each module looks up the name `Together`, so
every call goes through one place without changing a verifier:

    cache      get(model, prompt) / put(model, prompt, text); failures are
               never cached
    backend    complete(model, prompt, **decode) and chat(model, messages,
               **decode), both returning text (TogetherBackend)
    rewrite    optional rewrite(kind, prompt), the prompt to send instead
               (used by prompt_swap to send the v2 extraction prompts)
    failures   recorded (kind, model, error), then re-raised unchanged

The cache key is (model, prompt); a chat call's prompt is its message list as
canonical JSON. Decode settings are not in the key, since each HALoGEN call
site sends fixed ones (extraction.DECODING).
"""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Callable, Iterable, List, Optional


def chat_prompt(messages) -> str:
    """The cache prompt for a chat call: the messages as canonical JSON."""
    return json.dumps(list(messages), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


class TogetherBackend:
    """Together AI, the provider HALoGEN was written against."""

    def __init__(self, api_key: str):
        from together import Together

        self._client = Together(api_key=api_key)

    def complete(self, model: str, prompt: str, **decode) -> str:
        out = self._client.completions.create(model=model, prompt=prompt, **decode)
        return out.choices[0].text if out.choices else ""

    def chat(self, model: str, messages, **decode) -> str:
        out = self._client.chat.completions.create(model=model, messages=messages, **decode)
        return out.choices[0].message.content if out.choices else ""


class _Completions:
    def __init__(self, owner: "LLMClient"):
        self._owner = owner

    def create(self, model=None, prompt=None, **decode):
        if self._owner.rewrite is not None:
            prompt = self._owner.rewrite("completion", prompt)
        text = self._owner._call(
            "completion", model, prompt,
            lambda: self._owner.backend.complete(model, prompt, **decode))
        return SimpleNamespace(choices=[SimpleNamespace(text=text)])


class _ChatCompletions:
    def __init__(self, owner: "LLMClient"):
        self._owner = owner

    def create(self, model=None, messages=None, **decode):
        text = self._owner._call(
            "chat", model, chat_prompt(messages),
            lambda: self._owner.backend.chat(model, messages, **decode))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


class LLMClient:
    """Stands in for `together.Together` inside HALoGEN's verifiers."""

    def __init__(self, backend, cache=None, rewrite=None):
        self.backend = backend
        self.cache = cache
        # applied before the cache lookup, so the cache key is what was sent
        self.rewrite = rewrite
        self.completions = _Completions(self)
        self.chat = SimpleNamespace(completions=_ChatCompletions(self))
        self.calls = 0
        self.hits = 0
        self.failures: List[dict] = []

    def _call(self, kind: str, model: str, prompt: str, send: Callable[[], str]) -> str:
        if self.cache is not None:
            got = self.cache.get(model, prompt)
            if got is not None:
                self.hits += 1
                return got
        self.calls += 1
        try:
            text = send()
        except Exception as e:  # noqa: BLE001
            self.failures.append({"kind": kind, "model": model,
                                  "error": "%s: %s" % (type(e).__name__, e)})
            raise
        text = "" if text is None else str(text)
        if self.cache is not None:
            self.cache.put(model, prompt, text)
        return text

    def stats(self) -> dict:
        return {"calls": self.calls, "cache_hits": self.hits,
                "failures": len(self.failures)}


# summarization and simplification reach Together through this helper
HELPER_MODULES = ("get_model_responses", "verifiers.get_model_responses")


def route(modules: Iterable[Optional[ModuleType]], client: LLMClient) -> Callable[[], None]:
    """Make `Together(...)` in each module return `client`. Returns an undo.

    Only modules that bind the name `Together` are changed.
    """
    def factory(*args, **kwargs):
        return client

    targets = [m for m in modules if m is not None]
    targets += [sys.modules[n] for n in HELPER_MODULES if n in sys.modules]
    saved = []
    for m in targets:
        if "Together" in vars(m):
            saved.append((m, vars(m)["Together"]))
            m.Together = factory

    def undo() -> None:
        for m, original in saved:
            m.Together = original

    return undo
