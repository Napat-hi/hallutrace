"""hallutrace/fetch_sampling.py  |  which occurrences to download (pure, no network)

Per (index, query string) the pool is the plain and glued occurrences together
(they are disjoint token sequences): cnt = cnt_plain + cnt_glued. If cnt <=
CAP_FULL, fetch every occurrence of both forms. Otherwise sample SAMPLE_N
uniformly at random over ALL occurrences of both forms across shards, with a
seeded RNG.

The sample size is a flat SAMPLE_N, never "sample more if nothing matched":
sampling that depends on the outcome biases the estimate. Tested 29 Sep on
Oberto Pallavicino's 159 windows: a random 20 missed the fact 28% and 40% of the
time, a random 50 missed it 3% and 7%.

No rank list is ever built: indices are drawn in [0, cnt) and mapped to
(form, shard, rank) through cumulative segment sizes, so a name with 3M
mentions costs 50 draws, not 3M ranks.
"""

from __future__ import annotations

import bisect
import hashlib
import random

CAP_FULL = 200
SAMPLE_N = 50
SEED = 20260929

# Pilot buckets by OLMo 2 mention count (inclusive bounds); None = unbounded.
BUCKETS = ((0, 0), (1, 10), (11, 100), (101, 1000), (1001, None))
PER_BUCKET = 4

# Plain sorts before glued. Not alphabetical ("glued" < "plain"): it must match
# the pool order used in select_positions.
_FORM_ORDER = {"plain": 0, "glued": 1}


def derive_seed(base_seed: int, *parts) -> int:
    """A stable per-(index, query string) seed from `base_seed` and `parts`
    (sha256, NOT hash(), which changes between processes)."""
    # The ASCII unit separator cannot appear in an index name or a query string,
    # so ("ab", "c") and ("a", "bc") cannot collide.
    payload = "\x1f".join([str(base_seed)] + [str(p) for p in parts])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _sizes(segments: list[list[int]], form: str) -> list[int]:
    sizes = []
    for seg in segments:
        # Anything that is not a [start, end) pair means the find response was
        # misread; failing loudly beats silently sampling the wrong ranks.
        if len(seg) != 2 or seg[1] < seg[0]:
            raise ValueError("malformed %s segment %r" % (form, seg))
        sizes.append(seg[1] - seg[0])
    return sizes


def select_positions(
    segments_plain: list[list[int]],
    segments_glued: list[list[int]],
    cap_full: int,
    sample_n: int,
    seed: int,
) -> list[tuple[str, int, int]]:
    """[(form, shard, rank), ...] to fetch, sorted by (form, shard, rank).

    `segments_*` are infini-gram's segment_by_shard: one [start, end) per shard,
    occurrences in shard s are ranks start..end-1 ("plain" pool first, then
    "glued", in shard order). All positions if the combined count <= cap_full,
    else exactly sample_n distinct positions drawn uniformly without replacement
    with random.Random(seed).
    """
    sizes = _sizes(segments_plain, "plain") + _sizes(segments_glued, "glued")
    total = sum(sizes)
    if total == 0:
        return []

    # One flat pool: plain shards in order, then glued shards in order.
    pool = [("plain", s, seg) for s, seg in enumerate(segments_plain)]
    pool += [("glued", s, seg) for s, seg in enumerate(segments_glued)]

    if total <= cap_full or sample_n >= total:
        indices = range(total)
    else:
        # range is lazy, so sampling from millions of occurrences never builds
        # a list of them.
        indices = random.Random(seed).sample(range(total), sample_n)

    # ends[i] = number of pool indices covered by segments 0..i. An empty
    # segment repeats the previous value, so bisect_right skips over it.
    ends = []
    running = 0
    for n in sizes:
        running += n
        ends.append(running)

    out = []
    for idx in indices:
        i = bisect.bisect_right(ends, idx)
        form, shard, seg = pool[i]
        out.append((form, shard, seg[0] + idx - (ends[i] - sizes[i])))
    out.sort(key=lambda t: (_FORM_ORDER[t[0]], t[1], t[2]))
    return out


def _bucket_of(cnt: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if cnt >= lo and (hi is None or cnt <= hi):
            return i
    raise ValueError("count %r fits no pilot bucket" % (cnt,))


def choose_pilot(
    counts_by_name: dict[str, int], seed: int, per_bucket: int = PER_BUCKET
) -> list[str]:
    """About 20 names stratified over BUCKETS by OLMo 2 mention count (the
    combined plain + glued cnt of the main name): per_bucket from each bucket, or
    all of a smaller one. Seeded and deterministic; returned sorted by
    (bucket, name)."""
    members = [[] for _ in BUCKETS]
    for name, cnt in counts_by_name.items():
        members[_bucket_of(cnt)].append(name)

    picked = []
    for b, names in enumerate(members):
        # Sorting first makes the draw independent of dict insertion order.
        names.sort()
        if len(names) > per_bucket:
            names = random.Random(derive_seed(seed, b)).sample(names, per_bucket)
        picked.extend((b, n) for n in names)
    picked.sort()
    return [n for _, n in picked]
