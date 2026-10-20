"""Tests for hallutrace/cache.py. No key, no network, no real sleep.

Follows the pattern in tests/test_semantic_scholar.py: plain check()/eq(),
a fake transport for the end-to-end test, no unittest, no pytest.
"""

import os
import sys
import tempfile

from hallutrace.cache import (
    SemanticScholarCache,
    TogetherCache,
    connect_cache,
    init_cache_db,
    together_cache_key,
)
from hallutrace.semantic_scholar import (
    NOT_FOUND,
    UNVERIFIABLE,
    SemanticScholarClient,
    normalise,
)

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


def fresh_cache():
    conn = connect_cache(":memory:")
    init_cache_db(conn)
    return conn


print("1. the together_cache key recipe is frozen")
eq(
    together_cache_key(
        "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "Decompose this response into atomic units: Ada Lovelace was born in 1815.",
    ),
    "eae81d5800322d505aa8fc3ea8c6ad738283f49b99a859997d7264edb40e9a3e",
    "a known (model, prompt) pair hashes to a known digest; if this ever "
    "fails, the recipe changed and every row already cached is now an "
    "unreachable, silent miss",
)
eq(len(together_cache_key("m", "p")), 64, "sha256 hex digest is 64 characters")
check(
    together_cache_key("a", "bc") != together_cache_key("ab", "c"),
    "the \\x1f separator keeps 'a'+'bc' from colliding with 'ab'+'c'",
)

print()
print("2. WAL mode and a busy timeout, on a real file (WAL needs one, not :memory:)")
tmpdir = tempfile.mkdtemp(prefix="hallutrace_cache_test_")
try:
    db_path = os.path.join(tmpdir, "api_cache.sqlite")
    conn = connect_cache(db_path)
    init_cache_db(conn)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    eq(mode.lower(), "wal", "journal_mode is WAL")
    timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    check(timeout >= 1000, "busy_timeout is set to at least one second, got %d ms" % timeout)
    conn.close()
finally:
    import shutil

    shutil.rmtree(tmpdir, ignore_errors=True)

print()
print("3. init_cache_db is idempotent")
conn = fresh_cache()
init_cache_db(conn)  # second call must not raise
check(True, "calling init_cache_db twice does not raise")

print()
print("4. SemanticScholarCache: the get/put contract semantic_scholar.py expects")
conn = fresh_cache()
s2 = SemanticScholarCache(conn)
eq(s2.get("Attention Is All You Need"), None, "a miss returns None")
s2.put("Attention Is All You Need", "Attention Is All You Need")
eq(s2.get("attention is ALL you need"), "Attention Is All You Need",
   "a differently-cased title is the same key, via normalise()")
eq(s2.get("  Attention   is all you need  "), "Attention Is All You Need",
   "whitespace differences are the same key too")

s2.put("A missing paper", NOT_FOUND)
eq(s2.get("A missing paper"), NOT_FOUND, "NOT_FOUND is written, because it is an answer")

s2.put("Something transient", UNVERIFIABLE)
eq(s2.get("Something transient"), None,
   "UNVERIFIABLE is never persisted, even though put() was called directly "
   "with it; a transient failure must not become a permanent verdict")

print()
print("5. TogetherCache: the same shape for (model, prompt) -> response")
conn = fresh_cache()
tc = TogetherCache(conn)
eq(tc.get("olmo2-7b", "some prompt"), None, "a miss returns None")
tc.put("olmo2-7b", "some prompt", "a cached response")
eq(tc.get("olmo2-7b", "some prompt"), "a cached response", "a hit returns the cached response")
eq(tc.get("olmo2-7b", "a different prompt"), None,
   "a different prompt for the same model is a different key")
eq(tc.get("dclm-7b", "some prompt"), None,
   "the same prompt for a different model is a different key")

print()
print("6. end to end: a rerun over cached titles makes zero API calls")


class Reply:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = {}

    def json(self):
        return self._payload


class Transport:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers=None, params=None):
        self.calls.append(params)
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]


conn = fresh_cache()
cache = SemanticScholarCache(conn)
hit = Reply(200, {"data": [{"title": "Attention Is All You Need"}]})

transport1 = Transport([hit])
client1 = SemanticScholarClient(api_key="k", cache=cache, transport=transport1, sleep=lambda s: None)
eq(client1.title_for("attention is all you need"), "Attention Is All You Need",
   "first run resolves via the transport")
eq(len(transport1.calls), 1, "first run made one request")

transport2 = Transport([hit])
client2 = SemanticScholarClient(api_key="k", cache=cache, transport=transport2, sleep=lambda s: None)
eq(client2.title_for("Attention is all you need"), "Attention Is All You Need",
   "a fresh client sharing the cache still resolves the title")
eq(len(transport2.calls), 0, "a rerun over a cached title makes zero API calls")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
