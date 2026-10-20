"""Tests for hallutrace/llm_client.py. No key, no network, no Together import."""

import os
import sys
import tempfile
import types

from hallutrace import cache as hcache
from hallutrace import llm_client as lc

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


class Backend:
    def __init__(self):
        self.sent = []
        self.raise_next = None
        self.reply = "answer"

    def complete(self, model, prompt, **decode):
        self.sent.append(("completion", model, prompt, decode))
        if self.raise_next:
            e, self.raise_next = self.raise_next, None
            raise e
        return self.reply

    def chat(self, model, messages, **decode):
        self.sent.append(("chat", model, messages, decode))
        return self.reply


class DictCache:
    def __init__(self):
        self.d = {}

    def get(self, model, prompt):
        return self.d.get((model, prompt))

    def put(self, model, prompt, text):
        self.d[(model, prompt)] = text


M = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
MSG = [{"role": "user", "content": "Is this true?"}]

print("1. the client answers the way Together's client does")
b = Backend()
c = lc.LLMClient(b)
out = c.completions.create(model=M, prompt="p", max_tokens=300, stop=["<Context>:"])
check(out.choices[0].text == "answer", "completions: choices[0].text")
check(b.sent[0][3] == {"max_tokens": 300, "stop": ["<Context>:"]},
      "decode settings reach the backend unchanged")
out = c.chat.completions.create(model=M, messages=MSG)
check(out.choices[0].message.content == "answer", "chat: choices[0].message.content")
check(b.sent[1][2] == MSG, "the message list reaches the backend unchanged")
b.reply = None
check(c.completions.create(model=M, prompt="q").choices[0].text == "",
      "a None reply becomes an empty string, as HALoGEN's callers expect")
check(c.stats() == {"calls": 3, "cache_hits": 0, "failures": 0}, "stats() counts paid calls")

print("2. the cache")
b, cache = Backend(), DictCache()
c = lc.LLMClient(b, cache)
c.completions.create(model=M, prompt="p")
c.completions.create(model=M, prompt="p")
check(len(b.sent) == 1 and c.stats()["cache_hits"] == 1, "the same (model, prompt) is paid for once")
c.completions.create(model="other/model", prompt="p")
check(len(b.sent) == 2, "a different model is a different entry")
c.chat.completions.create(model=M, messages=[{"role": "user", "content": "p"}])
check(len(b.sent) == 3, "a chat call never hits a completion's entry, even with the same text")
check((M, lc.chat_prompt([{"role": "user", "content": "p"}])) in cache.d,
      "a chat call is keyed on its messages as canonical JSON")
check(lc.chat_prompt([{"content": "x", "role": "user"}]) == lc.chat_prompt([{"role": "user", "content": "x"}]),
      "key order inside a message does not change the key")
b.reply = ""
c.completions.create(model=M, prompt="empty")
c.completions.create(model=M, prompt="empty")
check(len(b.sent) == 4, "an empty reply is a real answer and is cached too")

print("3. failures are counted, re-raised, and never cached")
b, cache = Backend(), DictCache()
c = lc.LLMClient(b, cache)
b.raise_next = TimeoutError("read timed out")
try:
    c.completions.create(model=M, prompt="p")
    check(False, "the failure is re-raised")
except TimeoutError:
    check(True, "the failure is re-raised unchanged, so each verifier behaves as before")
check(len(c.failures) == 1 and c.failures[0] == {
    "kind": "completion", "model": M, "error": "TimeoutError: read timed out"},
    "the failure is kept with its kind, model and error")
check(cache.d == {}, "nothing was cached for it")
c.completions.create(model=M, prompt="p")
check(len(b.sent) == 2 and cache.d, "the retry is paid for, then cached")

print("4. the real API cache works as the cache")
tmp = tempfile.mkdtemp()
conn = hcache.connect_cache(os.path.join(tmp, "api_cache.sqlite"))
hcache.init_cache_db(conn)
b = Backend()
c = lc.LLMClient(b, hcache.TogetherCache(conn))
c.completions.create(model=M, prompt="p")
c2 = lc.LLMClient(Backend(), hcache.TogetherCache(conn))
check(c2.completions.create(model=M, prompt="p").choices[0].text == "answer"
      and c2.stats()["cache_hits"] == 1, "a new client on the same file is served from it")

print("5. route() puts the client where each module looks up Together")
original = object()
uses = types.ModuleType("fake_verifier")
uses.Together = original
other = types.ModuleType("fake_other")
helper = types.ModuleType("get_model_responses")
helper.Together = original
sys.modules["get_model_responses"] = helper
c = lc.LLMClient(Backend())
undo = lc.route([uses, other, None], c)
check(uses.Together(api_key="k") is c, "a module that binds Together now builds the client")
check(helper.Together() is c, "get_model_responses is routed too, since two verifiers go through it")
check(not hasattr(other, "Together"), "a module that never binds Together is left alone")
undo()
check(uses.Together is original and helper.Together is original, "undo() restores every module")
del sys.modules["get_model_responses"]

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all tests passed")
