"""Tests for hallutrace/semantic_scholar.py. No network, no key, no real sleep."""

import sys

from hallutrace.semantic_scholar import (
    NOT_FOUND,
    UNVERIFIABLE,
    NullCache,
    SemanticScholarClient,
    normalise,
)


class FakeCache:
    """Stands in for a cache: two methods, keys normalised."""

    def __init__(self):
        self.data = {}

    def get(self, title):
        return self.data.get(normalise(title))

    def put(self, title, verdict):
        self.data[normalise(title)] = verdict

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


class Reply:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._payload


class Transport:
    """Returns the queued replies in order, then repeats the last one."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, url, headers=None, params=None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]


class Clock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


def client(replies, **kw):
    c = Clock()
    t = Transport(replies)
    return (
        SemanticScholarClient(
            api_key="test-key", transport=t, sleep=c.sleep, clock=c.now, **kw
        ),
        t,
        c,
    )


hit = Reply(200, {"data": [{"title": "Attention Is All You Need"}]})
miss = Reply(200, {"data": []})

print("1. normalisation")
eq(normalise("  The  ETHICS of Information "), "the ethics of information",
   "case and whitespace collapse")
eq(normalise("Bogotá Studies"), "bogota studies", "accents fold")

print("2. the three verdicts")
cl, tr, _ = client([hit])
eq(cl.title_for("attention is all you need"), "Attention Is All You Need",
   "a found paper returns its canonical title")
cl, tr, _ = client([miss])
eq(cl.title_for("A paper that does not exist"), NOT_FOUND,
   "an empty data list is NOT_FOUND, which counts as hallucinated")
cl, tr, _ = client([Reply(403)])
eq(cl.title_for("x"), UNVERIFIABLE, "403 is UNVERIFIABLE, not a hallucination")
eq(cl.title_for(None), UNVERIFIABLE, "a non string is UNVERIFIABLE")
eq(cl.title_for("   "), UNVERIFIABLE, "a blank title is UNVERIFIABLE")

print("3. one request per second")
cl, tr, ck = client([hit, hit, hit])
for t in ("a", "b", "c"):
    cl.title_for(t)
eq(len(tr.calls), 3, "three distinct titles make three requests")
check(all(abs(s - 1.0) < 1e-9 for s in ck.slept),
      "each request after the first waits out the remaining second")
eq(len(ck.slept), 2, "the first request does not wait")

print("4. 429 is a normal event, not a failure")
cl, tr, ck = client([Reply(429), Reply(429), hit])
eq(cl.title_for("attention"), "Attention Is All You Need",
   "two 429s then success still resolves")
eq(cl.stats["retries"], 2, "both 429s counted as retries")
check(len(ck.slept) >= 2, "it backed off between attempts")
backoffs = [s for s in ck.slept if s > 1.0]
check(len(backoffs) >= 2 and backoffs[1] > backoffs[0],
      "the backoff grows, %r" % backoffs)

cl, tr, ck = client([Reply(429, headers={"Retry-After": "7"}), hit])
cl.title_for("attention")
check(7.0 in ck.slept, "Retry-After is honoured when the server sends one")

cl, tr, ck = client([Reply(429)], max_retries=3)
eq(cl.title_for("x"), UNVERIFIABLE,
   "429 forever gives up as UNVERIFIABLE, never as a hallucination")
eq(len(tr.calls), 3, "it gave up after max_retries attempts")

print("5. a transport that raises is retried, not propagated")


class Boom:
    def __init__(self):
        self.n = 0

    def __call__(self, url, headers=None, params=None):
        self.n += 1
        if self.n < 3:
            raise OSError("connection reset")
        return hit


ck = Clock()
cl = SemanticScholarClient(api_key="k", transport=Boom(), sleep=ck.sleep, clock=ck.now)
eq(cl.title_for("attention"), "Attention Is All You Need",
   "two dropped connections then success still resolves")

print("6. the cache seam")
check(isinstance(SemanticScholarClient(api_key="k").cache, NullCache),
      "with no cache given, the client still works and remembers nothing")

cache = FakeCache()
cl, tr, _ = client([hit], cache=cache)
cl.title_for("Attention Is All You Need")
cl.title_for("  attention is ALL you need  ")
eq(len(tr.calls), 1, "a differently cased repeat is a cache hit")
eq(cl.stats["hits"], 1, "the hit is counted")

cl, tr, _ = client([miss], cache=cache)
cl.title_for("A paper that does not exist")
eq(cache.data[normalise("A paper that does not exist")], NOT_FOUND,
   "NOT_FOUND is written, because it is an answer")

cl, tr, _ = client([Reply(500)], cache=cache, max_retries=1)
eq(cl.title_for("Something transient"), UNVERIFIABLE, "500 is UNVERIFIABLE")
check(normalise("Something transient") not in cache.data,
      "UNVERIFIABLE is NOT written, so the next run retries it")

cl2, tr2, _ = client([hit], cache=cache)
eq(cl2.title_for("attention is all you need"), "Attention Is All You Need",
   "a second client sharing the cache reuses the verdict")
eq(len(tr2.calls), 0, "and makes no request for it")

print("7. the request itself")
cl, tr, _ = client([hit])
cl.title_for("attention")
call = tr.calls[0]
eq(call["params"]["fields"], "title",
   "asks for the title in one call, not a url then a second lookup")
eq(call["headers"].get("x-api-key"), "test-key", "the key is sent as a header")
check("query" in call["params"], "the title goes in query, not in the path")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
