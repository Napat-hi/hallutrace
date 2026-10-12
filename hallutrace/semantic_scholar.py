"""Semantic Scholar client: one request per second, backoff, three verdicts.

`/paper/search` allows one request per second even with a key, and returns 429
often. This client paces itself from the last request, retries 429 and 5xx with
exponential backoff and jitter (honouring Retry-After), and reads and writes
through any cache with get(title) and put(title, verdict).

    a title string   the API found a paper
    NOT_FOUND        the API answered and found no such paper
    UNVERIFIABLE     the lookup failed

The released code returned '' for both of the last two, so a network failure
was scored as a hallucination. UNVERIFIABLE is never cached, so the next run
retries it.
"""

from __future__ import annotations

import os
import random
import re
import threading
import time
import unicodedata
from typing import Callable, Optional

UNVERIFIABLE = "__UNVERIFIABLE__"
NOT_FOUND = "__NOT_FOUND__"

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# /paper/search allows 1 request per second, even with a key
MIN_INTERVAL = 1.0
MAX_RETRIES = 5
BASE_BACKOFF = 2.0
MAX_BACKOFF = 60.0
RETRYABLE = (429, 500, 502, 503, 504)


def normalise(title: str) -> str:
    """Cache key: case, accents and runs of whitespace removed."""
    s = unicodedata.normalize("NFKD", str(title))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip().lower()


class NullCache:
    """A cache that remembers nothing.

    A real cache implements get(title) -> verdict or None and
    put(title, verdict), with keys compared by normalise().
    """

    def get(self, title):
        return None

    def put(self, title, verdict):
        pass


class SemanticScholarClient:
    """Paced, retrying lookups against /paper/search.

    `transport` (default requests.get) takes (url, headers, params) and returns
    an object with status_code, json() and headers; tests replace it.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache=None,
        min_interval: float = MIN_INTERVAL,
        max_retries: int = MAX_RETRIES,
        transport: Optional[Callable] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("S2_API_KEY", "")
        self.cache = cache if cache is not None else NullCache()
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.transport = transport
        self._sleep = sleep
        self._clock = clock
        self._last_request = None
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "requests": 0, "retries": 0, "unverifiable": 0}

    # -- pacing -------------------------------------------------------------

    def _wait_turn(self):
        if self._last_request is None:
            return
        gap = self._clock() - self._last_request
        if gap < self.min_interval:
            self._sleep(self.min_interval - gap)

    @staticmethod
    def _retry_after(response, attempt):
        header = getattr(response, "headers", None) or {}
        raw = header.get("Retry-After") or header.get("retry-after")
        if raw:
            try:
                return min(float(raw), MAX_BACKOFF)
            except (TypeError, ValueError):
                pass
        delay = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
        return delay * (0.5 + random.random() / 2.0)  # jitter, 50% to 100%

    # -- the one public call ------------------------------------------------

    def title_for(self, title) -> str:
        """The canonical title, NOT_FOUND, or UNVERIFIABLE. Never raises."""
        if not title or not isinstance(title, str) or not title.strip():
            return UNVERIFIABLE

        cached = self.cache.get(title)
        if cached is not None:
            self.stats["hits"] += 1
            return cached

        verdict = self._lookup(title)
        if verdict == UNVERIFIABLE:
            self.stats["unverifiable"] += 1
        else:
            self.cache.put(title, verdict)
        return verdict

    def _lookup(self, title) -> str:
        get = self.transport
        if get is None:
            import requests

            get = requests.get

        headers = {"x-api-key": self.api_key} if self.api_key else {}
        params = {"query": title, "fields": "title", "limit": 1}

        for attempt in range(self.max_retries):
            with self._lock:
                self._wait_turn()
                self._last_request = self._clock()
                self.stats["requests"] += 1
                try:
                    response = get(SEARCH_URL, headers=headers, params=params)
                except Exception:
                    response = None

            if response is None:
                self.stats["retries"] += 1
                self._sleep(self._retry_after(None, attempt))
                continue

            code = getattr(response, "status_code", 0)

            if code == 200:
                try:
                    payload = response.json()
                except Exception:
                    return UNVERIFIABLE
                data = payload.get("data") or []
                if not data:
                    return NOT_FOUND
                found = data[0].get("title")
                return found if found else NOT_FOUND

            if code in RETRYABLE:
                self.stats["retries"] += 1
                self._sleep(self._retry_after(response, attempt))
                continue

            # other errors (400, 401, 403): retrying will not help
            return UNVERIFIABLE

        return UNVERIFIABLE

    def close(self):
        save = getattr(self.cache, "save", None)
        if callable(save):
            save()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
