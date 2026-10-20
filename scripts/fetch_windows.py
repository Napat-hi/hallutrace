"""Fetch text windows around every corpus mention of the task's subjects.

For each subject name: find every mention in each subject model's training corpus
(infini-gram), download a window of text around each one (or around a seeded
sample when there are too many), keep everything in a SQLite cache. Matching and
counting read from the cache later; they are not here.

    plan    store the query strings, tokenize their glued forms, run the finds on
            OLMo 2 only, and choose the pilot names once. No windows.
    pilot   finds + windows for the pilot names on all five indexes.
    full    finds + windows for every name on all five indexes, OLMo 2 first.
    status  what is done and planned, errors, ETA. Never touches the network.

Everything resumes for free: anything already in the cache is skipped, so a
killed run, --max-requests or a 403 stop loses nothing. --dry-run sends nothing
and does not write to the cache.

    .venv\\Scripts\\python.exe scripts\\fetch_windows.py plan --dry-run
    .venv\\Scripts\\python.exe scripts\\fetch_windows.py plan
    .venv\\Scripts\\python.exe scripts\\fetch_windows.py pilot
    .venv\\Scripts\\python.exe scripts\\fetch_windows.py full
    .venv\\Scripts\\python.exe scripts\\fetch_windows.py status

A task can be split between fetchers on different machines: `--shard i/n` keeps
every n-th name of the sorted list, starting at the i-th, so all query strings of one
name stay in one shard. Each shard has its own cache, which records its shard and
refuses a plan for another one; a shard holder runs `plan --task biographies --shard
2/3`, then `full --task biographies` on that cache.

The driver only talks to the network through hallutrace.infinigram.InfinigramClient,
which owns the rate limit (1.05 s between request starts, strictly sequential),
the retry policy and the lock file. No threads, no asyncio, no multiprocessing.

Exit codes: 0 done, or a clean stop at --max-requests, with nothing failed for
good in this run; 1 other error (including a refused constants change); 2 another
fetcher holds the lock; 3 blocked by infini-gram (3 blocks in a row: HTTP 403 or
429); 4 too many consecutive failed requests; 5 the run finished but some items
are unfetched (rerun to retry); 130 Ctrl+C.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from hallutrace import fetch_names  # noqa: E402
from hallutrace import infinigram as _infinigram  # noqa: E402
from hallutrace.claim_key import AliasTable  # noqa: E402
from hallutrace.fetch_names import QueryString  # noqa: E402
from hallutrace.fetch_sampling import (  # noqa: E402
    CAP_FULL,
    SAMPLE_N,
    SEED,
    choose_pilot,
    derive_seed,
    select_positions,
)
from hallutrace.infinigram import (  # noqa: E402
    DISP_LEN,
    INDEXES,
    MIN_GAP_S,
    Blocked,
    Cache,
    HttpTransport,
    InfinigramClient,
    LockHeld,
    MaxRequestsReached,
    Throttle,
)

# Raised by the client after MAX_CONSECUTIVE_FAILURES failed requests in a row.
# A placeholder that is never raised keeps this driver importable until it lands.
TooManyFailures = getattr(_infinigram, "TooManyFailures", None) or type("TooManyFailures", (Exception,), {})

SCRIPT_VERSION = "1"

DATA_BASE = ROOT
DEFAULT_PROMPTS = DATA_BASE / "data" / "prompts.csv"
DEFAULT_ALIAS_TABLE = DATA_BASE / "data" / "alias_table.json"
DEFAULT_SNAPSHOT = DATA_BASE / "data" / "alias_build" / "wikidata_snapshot.json"
DEFAULT_CACHE = DATA_BASE / "data" / "corpus_cache" / "infinigram.sqlite"

OLMO2 = INDEXES[0]  # `plan` and the first `full` pass run on it alone
DEFAULT_TASK = "historical_events"

# Task registry: task name -> loader of the unique prompt names. Nothing else in the
# driver knows about a specific task.
TASKS = {
    "historical_events": lambda prompts_path: fetch_names.load_names(prompts_path, "historical_events"),
    "biographies": lambda prompts_path: fetch_names.load_names(prompts_path, "biographies"),
}


def parse_shard(spec):
    """"2/3" -> (2, 3). None stays None (the whole task)."""
    if spec is None:
        return None
    try:
        i, n = (int(x) for x in spec.split("/"))
    except ValueError:
        raise UsageError("--shard must look like i/n, e.g. 2/3, not %r" % spec)
    if not (n >= 1 and 1 <= i <= n):
        raise UsageError("--shard %s: need 1 <= i <= n" % spec)
    return i, n


def shard_names(names, shard):
    """Every n-th name of the sorted list, from the i-th: deterministic, and one name's
    query strings never straddle two shards. A query string shared by two names in
    different shards is fetched by both; the merge counts it once."""
    if shard is None:
        return sorted(names)
    i, n = shard
    return [name for k, name in enumerate(sorted(names)) if k % n == i - 1]


def shard_label(shard):
    return "all" if shard is None else "%d/%d" % shard


def check_shard(cache, args, shard):
    """Refuse a plan for another shard than the one this cache holds, or a shard of a
    cache planned whole: either would deactivate names and mix shards in one cache."""
    stored = cache.get_meta(tkey("shard", args.task))
    if stored is None:
        if shard is not None and cache.get_meta(tkey("task_names", args.task)) is not None:
            raise UsageError("this cache was planned for the whole task %s; use a separate --cache "
                             "for shard %s" % (args.task, shard_label(shard)))
        return
    if args.command == "plan" and stored != shard_label(shard):
        raise UsageError("this cache holds shard %s of %s, not %s; use a separate --cache per shard"
                         % (stored, args.task, shard_label(shard)))
    if args.command != "plan" and shard is not None and stored != shard_label(shard):
        raise UsageError("this cache holds shard %s of %s, not %s" % (stored, args.task, shard_label(shard)))

# Keys of the three input files in run_meta (input_sha256_<k>, input_path_<k>).
INPUTS = ("prompts", "alias_table", "snapshot")
INPUT_DEFAULTS = {"prompts": DEFAULT_PROMPTS, "alias_table": DEFAULT_ALIAS_TABLE, "snapshot": DEFAULT_SNAPSHOT}

# A string whose `count` (tokenize) call failed for good this many runs is used
# plain only, like a string whose tokens lack the "(" prefix. One attempt per
# run, so this is also the number of runs it takes to give up.
TOKENIZE_MAX_FAILS = 3

PROGRESS_EVERY = 200  # requests between progress lines
FALLBACK_RATE_PER_H = 3200  # measured 29 Sep on OLMo 2 at the 1.05 s gap
RATE_MIN_SAMPLES = 20  # fewer timestamps than this and the observed rate is noise
RATE_MAX_GAP_S = 120.0  # a longer pause between two results is idle time, not rate
RATE_WINDOW_ROWS = 2000  # "recent stretch of work": the newest this-many results


def say(msg="", err=False):
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def tkey(base, task):
    """run_meta key for a per-task value. The default task keeps the bare key."""
    return base if task == DEFAULT_TASK else "%s:%s" % (base, task)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info():
    """(HEAD hash, dirty flag) of the repo this script lives in, 'unknown' on any failure.
    --no-optional-locks: recording provenance must never write to the repo."""
    def run(*a):
        return subprocess.run(
            ["git", "--no-optional-locks", *a], cwd=str(ROOT), capture_output=True,
            text=True, timeout=20, check=True,
        ).stdout
    try:
        head = run("rev-parse", "HEAD").strip()
        dirty = "true" if run("status", "--porcelain").strip() else "false"
        return head, dirty
    except (OSError, subprocess.SubprocessError):
        return "unknown", "unknown"


def constants_now():
    return dict(CAP_FULL=CAP_FULL, SAMPLE_N=SAMPLE_N, DISP_LEN=DISP_LEN, MIN_GAP_S=MIN_GAP_S, SEED=SEED)


def constants_diff(cache):
    """{name: (stored, now)} for each constant recorded by an earlier `plan` that differs now."""
    raw = cache.get_meta("constants")
    if raw is None:
        return {}
    try:
        stored = json.loads(raw)
    except ValueError:
        return {}
    now = constants_now()
    return {k: (stored[k], now[k]) for k in now if k in stored and stored[k] != now[k]}


def describe_diff(diff):
    return ", ".join("%s: stored %s, now %s" % (k, a, b) for k, (a, b) in sorted(diff.items()))


def check_constants(cache, args):
    """Refuse to go on when the stored constants differ from this script's.

    The plan table and the fetched windows were chosen under the stored values
    (CAP_FULL decides all-vs-sample, SEED the sample, ...), so quietly mixing two
    settings in one cache would make the sample inconsistent.
    """
    diff = constants_diff(cache)
    if not diff:
        return
    if not getattr(args, "allow_constants_change", False):
        raise UsageError(
            "the constants stored in this cache's run_meta differ from this script's (%s). Windows "
            "and plan rows already cached were chosen under the stored values. Restore the constants, "
            "or rerun with --allow-constants-change to accept the new ones (run_meta is updated; "
            "nothing cached is refetched or reselected)." % describe_diff(diff))
    say("--allow-constants-change: run_meta constants updated (%s)" % describe_diff(diff))
    cache.set_meta("constants", json.dumps(constants_now()))


# -- glued forms ---------------------------------------------------------------
def tokenize_fails(cache, string):
    """How many runs' tokenize attempt for `string` failed for good (run_meta)."""
    raw = cache.get_meta("tokenize_fail:" + string)
    try:
        return int(raw) if raw is not None else 0
    except ValueError:
        return 0


def glued_state(cache, string):
    """('ok', ids) | ('failed', None) | ('unknown', None), from the cache alone.

    The glued form is the string tokenized after a "(" with that prefix dropped
    (fetch_names.glued_ids). 'failed' means that string simply has no glued pool:
    the tokens do not start with the "(" prefix (deterministic), or its tokenize
    request failed for good in TOKENIZE_MAX_FAILS runs (abandoned, plain only).
    'unknown' means the tokenize request has not succeeded yet, so the pool is
    undecidable this run and retried on the next.
    """
    hit = cache.get_glued_ids("(" + string)
    if hit is None:
        if tokenize_fails(cache, string) >= TOKENIZE_MAX_FAILS:
            return "failed", None
        return "unknown", None
    ids, tokens = hit
    try:
        return "ok", fetch_names.glued_ids(ids, tokens)
    except ValueError:
        return "failed", None


# -- the pool of one (index, query string), decided from cached finds ------------
def pool_finds(cache, index, string):
    """(plain, glued, needed, done, decided): the cached finds of the two forms.
    `decided` is True once every find the pool needs is cached."""
    state, _ = glued_state(cache, string)
    plain = cache.get_find(index, string, "plain")
    glued = cache.get_find(index, string, "glued") if state == "ok" else None
    needed = 1 + (0 if state == "failed" else 1)
    done = (plain is not None) + (glued is not None)
    decided = plain is not None and (state == "failed" or glued is not None)
    return plain, glued, needed, done, decided


def pool_cnt(cache, index, string):
    """Combined plain + glued cnt, or None while a find is missing."""
    plain, glued, _, _, decided = pool_finds(cache, index, string)
    if not decided:
        return None
    return plain["cnt"] + (glued["cnt"] if glued else 0)


def pool_positions(cache, index, string):
    """([(form, shard, rank), ...], total) from cached finds, or None if undecided.
    The one place positions are derived, so the live run, --dry-run and `status`
    can never disagree about what is planned."""
    plain, glued, _, _, decided = pool_finds(cache, index, string)
    if not decided:
        return None
    seg_glued = glued["segment_by_shard"] if glued else []
    positions = select_positions(
        plain["segment_by_shard"], seg_glued, CAP_FULL, SAMPLE_N,
        derive_seed(SEED, index, string),
    )
    total = sum(e - s for s, e in plain["segment_by_shard"]) + sum(e - s for s, e in seg_glued)
    return positions, total


def windows_present(cache, index, string):
    rows = cache.conn.execute(
        "SELECT form, s, rank FROM windows WHERE index_name = ? AND query_string = ?",
        (index, string),
    ).fetchall()
    return {(r["form"], r["s"], r["rank"]) for r in rows}


def store_plan(cache, index, string, positions, total):
    selected = "all" if total <= CAP_FULL else "sample"
    cache.conn.executemany(
        "INSERT OR IGNORE INTO plan (index_name, query_string, form, s, rank, selected) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [(index, string, form, s, rank, selected) for form, s, rank in positions],
    )
    cache.conn.commit()


# -- the entities table ------------------------------------------------------------
def build_universe(names, alias_table, snapshot):
    """{prompt name: [QueryString, ...]} from the input files (rules 1 to 4, 6)."""
    return {n: fetch_names.query_strings(n, alias_table, snapshot) for n in names}


def load_universe(cache, names):
    """{prompt name: [active QueryString, ...]} from the entities table."""
    order = {s: i for i, s in enumerate(fetch_names.SOURCES)}
    rows = cache.conn.execute(
        "SELECT prompt_name, query_string, source, qid, ambiguous_name, ambiguous_stripped "
        "FROM entities WHERE active = 1 ORDER BY rowid"
    ).fetchall()
    wanted = set(names)
    out = {n: [] for n in names}
    for r in rows:
        if r["prompt_name"] in wanted:
            out[r["prompt_name"]].append(QueryString(
                r["prompt_name"], r["query_string"], r["source"], r["qid"],
                bool(r["ambiguous_name"]), bool(r["ambiguous_stripped"]),
            ))
    for lst in out.values():
        lst.sort(key=lambda q: order.get(q.source, 99))  # stable: keeps insertion order within a source
    return out


def main_of(strings):
    for q in strings:
        if q.source == "name":
            return q
    return None


def sync_entities(cache, universe, task):
    """Bring the entities table in line with `universe`. Returns (added, deactivated, reactivated).

    New strings are inserted active. A string that is no longer produced is set
    active=0, never deleted, and its cached finds and windows are never touched,
    so a string that comes back is picked up again for free. Deactivation is
    scoped to this task's own names (this run's plus the previous run's), so a
    second task's rows are left alone.
    """
    conn = cache.conn
    existing = {
        (r["prompt_name"], r["query_string"]): r["active"]
        for r in conn.execute("SELECT prompt_name, query_string, active FROM entities")
    }
    current = {(q.prompt_name, q.string): q for lst in universe.values() for q in lst}
    prev = cache.get_meta(tkey("task_names", task))
    scope = set(json.loads(prev) if prev else []) | set(universe)
    added = deactivated = reactivated = 0
    for key, q in current.items():
        row = (q.source, q.qid, int(q.ambiguous_name), int(q.ambiguous_stripped))
        if key not in existing:
            conn.execute(
                "INSERT INTO entities (prompt_name, query_string, source, qid, ambiguous_name, "
                "ambiguous_stripped, active) VALUES (?, ?, ?, ?, ?, ?, 1)", (*key, *row))
            added += 1
        else:
            # Metadata is refreshed: an alias-table revision can change qid or the
            # ambiguity flags of a string that is otherwise unchanged.
            conn.execute(
                "UPDATE entities SET source = ?, qid = ?, ambiguous_name = ?, ambiguous_stripped = ?, "
                "active = 1 WHERE prompt_name = ? AND query_string = ?", (*row, *key))
            reactivated += existing[key] == 0
    for key, active in existing.items():
        if key[0] in scope and key not in current and active:
            conn.execute("UPDATE entities SET active = 0 WHERE prompt_name = ? AND query_string = ?", key)
            deactivated += 1
    conn.commit()
    return added, deactivated, reactivated


# -- the live pipeline ---------------------------------------------------------------
class Ctx:
    """Everything one live run shares: client, cache, counters."""

    def __init__(self, client, cache):
        self.client = client
        self.cache = cache
        self.hits = 0  # results served from the cache
        self.extra_sent = 0  # network requests spent on extra-form finds
        self.undecided = set()  # (index, name) whose main finds failed; extras skipped this run
        self.glued_reported = set()
        self.failed = 0  # requests that failed for good in this run: items left unfetched
        self.tok_tried = set()  # strings whose tokenize was attempted this run (one attempt per run)
        self.pool_memo = {}  # (index, string) -> combined cnt or None, so a shared string is asked once per run
        self.windows_run = set()  # (index, string) whose windows were attempted this run
        self._next_report = PROGRESS_EVERY

    def tick(self, index, name):
        """A progress line every PROGRESS_EVERY requests, so a long run is watchable."""
        sent = self.client.requests_sent
        if sent >= self._next_report:
            say("[%s] %s | %s: %d requests sent, %d cache hits"
                % (datetime.now().strftime("%H:%M:%S"), index, name, sent, self.hits))
            self._next_report = (sent // PROGRESS_EVERY + 1) * PROGRESS_EVERY

    def seen(self, result):
        if result is None:
            self.failed += 1  # logged in errors, retried on the next run
        elif result.get("from_cache"):
            self.hits += 1
        return result


def ensure_glued(ctx, string):
    """Glued ids of `string` (tokenized once, cached, index-independent), or None.
    None means no glued pool: either tokenization has not succeeded yet (retried
    next run) or the string is glued-failed: its tokens have no "(" prefix, or its
    tokenize request failed for good TOKENIZE_MAX_FAILS runs (reported, never
    fatal, plain form only).

    A failing tokenize is attempted once per run and counted in run_meta, so a
    string that can never be tokenized stops costing a request every time and
    stops blocking its pool.
    """
    state, ids = glued_state(ctx.cache, string)
    if state == "unknown" and string not in ctx.tok_tried:
        ctx.tok_tried.add(string)
        if ctx.client.tokenize("(" + string) is None:
            ctx.failed += 1
            n = tokenize_fails(ctx.cache, string) + 1
            ctx.cache.set_meta("tokenize_fail:" + string, str(n))
            if n >= TOKENIZE_MAX_FAILS:
                say("note: tokenizing %r failed %d runs in a row; its glued form is given up, plain only" % (string, n))
        state, ids = glued_state(ctx.cache, string)
    if state == "failed" and string not in ctx.glued_reported:
        ctx.glued_reported.add(string)
        if ctx.cache.get_glued_ids("(" + string) is not None:
            say("note: %r has no usable glued form (tokens lack the '(' prefix); plain form only" % string)
    return ids if state == "ok" else None


def find_pool(ctx, index, string):
    """Plain and glued finds of one query string. The combined cnt, or None when
    the pool is undecidable this run (a request failed for good). Asked once per
    (index, string) per run: a string shared by two prompt names is not resent."""
    key = (index, string)
    if key not in ctx.pool_memo:
        ctx.pool_memo[key] = _find_pool(ctx, index, string)
    return ctx.pool_memo[key]


def _find_pool(ctx, index, string):
    client = ctx.client
    plain = ctx.seen(client.find(index, string, "plain", query=string))
    if plain is None:
        return None
    ids = ensure_glued(ctx, string)
    state, _ = glued_state(ctx.cache, string)
    if state == "failed":
        return plain["cnt"]
    if ids is None:
        return None
    glued = ctx.seen(client.find(index, string, "glued", query_ids=ids))
    if glued is None:
        return None
    return plain["cnt"] + glued["cnt"]


def fetch_missing_windows(ctx, index, name, string):
    if (index, string) in ctx.windows_run:
        return
    ctx.windows_run.add((index, string))
    got = pool_positions(ctx.cache, index, string)
    if got is None:
        return
    positions, total = got
    store_plan(ctx.cache, index, string, positions, total)
    have = windows_present(ctx.cache, index, string)
    _, ids = glued_state(ctx.cache, string)
    for form, s, rank in positions:
        if (form, s, rank) in have:
            continue
        if form == "plain":
            r = ctx.client.get_doc_by_rank(index, string, "plain", s, rank, query=string, max_disp_len=DISP_LEN)
        else:
            r = ctx.client.get_doc_by_rank(index, string, "glued", s, rank, query_ids=ids, max_disp_len=DISP_LEN)
        ctx.seen(r)  # None: failed for good, logged, retried next run
        ctx.tick(index, name)


def process_name(ctx, index, name, strings, with_windows):
    """One (index, prompt name): finds before windows.

    1. the main name's finds, plain and glued;
    2. rule 5: the extra forms are queried on THIS index only if the main name's
       combined cnt is <= CAP_FULL (a famous name already has plenty of windows;
       extras only matter for recall where counts are small). If a main find
       failed the gate is undecidable: skip the extras now, retry next run;
    3. finds for each allowed extra;
    4. positions per (index, string) into the plan table;
    5. windows for every planned position not yet fetched.
    """
    main = main_of(strings)
    if main is None:
        return
    cnt = find_pool(ctx, index, main.string)
    ctx.tick(index, name)
    allowed = [main]
    if cnt is None:
        ctx.undecided.add((index, name))
    elif cnt <= CAP_FULL:
        before = ctx.client.requests_sent
        for q in strings:
            if q is main:
                continue
            find_pool(ctx, index, q.string)
            ctx.tick(index, name)
            allowed.append(q)
        ctx.extra_sent += ctx.client.requests_sent - before
    for q in allowed:
        got = pool_positions(ctx.cache, index, q.string)
        if got is not None:
            store_plan(ctx.cache, index, q.string, *got)
    if with_windows:
        for q in allowed:
            fetch_missing_windows(ctx, index, name, q.string)


def run_pipeline(ctx, universe, names, indexes, with_windows):
    # Index outer, names inner: OLMo 2 completes for every name before the next index.
    for index in indexes:
        for name in names:
            process_name(ctx, index, name, universe[name], with_windows)


# -- analysis from the cache alone (dry-run and status) -------------------------------
def analyse_index(cache, universe, names, index, with_windows):
    """Counts for one index from cached finds only; sends and writes nothing.

    finds_*: finds the pipeline needs, known so far (extras are only known for
    names whose main find is cached). windows_*: over the strings whose pool is
    decidable; `undecided_strings` are the rest, whose windows are unknown.
    """
    r = dict(finds_done=0, finds_missing=0, undecided_names=0, undecided_strings=0,
             windows_total=0, windows_done=0)
    # A string shared by two prompt names (two disambiguated names stripping to one
    # string) is fetched once, so it is counted once per (index, string).
    counted_finds, counted_windows = set(), set()

    def count_finds(string):
        if string in counted_finds:
            return
        counted_finds.add(string)
        _, _, needed, done, _ = pool_finds(cache, index, string)
        r["finds_done"] += done
        r["finds_missing"] += needed - done

    for name in names:
        main = main_of(universe[name])
        if main is None:
            continue
        cnt = None
        allowed = [main]
        count_finds(main.string)
        _, _, _, _, decided = pool_finds(cache, index, main.string)
        if decided:
            cnt = pool_cnt(cache, index, main.string)
        if cnt is None:
            r["undecided_names"] += 1
        elif cnt <= CAP_FULL:
            for q in universe[name]:
                if q is main:
                    continue
                count_finds(q.string)
                allowed.append(q)
        if with_windows:
            for q in allowed:
                if q.string in counted_windows:
                    continue
                counted_windows.add(q.string)
                got = pool_positions(cache, index, q.string)
                if got is None:
                    r["undecided_strings"] += 1
                    continue
                positions, _ = got
                have = windows_present(cache, index, q.string)
                r["windows_total"] += len(positions)
                r["windows_done"] += sum(1 for p in positions if p in have)
    r["windows_missing"] = r["windows_total"] - r["windows_done"]
    return r


def tokenize_needs(cache, universe, names):
    """(strings to tokenize, of which not yet cached) over the active strings of `names`."""
    seen = set()
    for n in names:
        for q in universe[n]:
            seen.add(q.string)
    missing = sum(1 for s in seen if glued_state(cache, s)[0] == "unknown")
    return len(seen), missing


def observed_rate(cache):
    """(results per second, samples) over the recent stretch of work, or None.

    Uses fetched_at of the newest finds and windows. Gaps above RATE_MAX_GAP_S are
    idle time between runs, not fetching speed, so they are left out. Tokenize
    calls carry no timestamp and are not counted.
    """
    rows = cache.conn.execute(
        "SELECT fetched_at FROM (SELECT fetched_at FROM finds UNION ALL SELECT fetched_at FROM windows) "
        "ORDER BY fetched_at DESC LIMIT ?", (RATE_WINDOW_ROWS,)).fetchall()
    try:
        times = sorted(datetime.fromisoformat(r["fetched_at"]).timestamp() for r in rows)
    except ValueError:
        return None
    gaps = [b - a for a, b in zip(times, times[1:]) if b - a <= RATE_MAX_GAP_S]
    if len(gaps) < RATE_MIN_SAMPLES or sum(gaps) <= 0:
        return None
    return len(gaps) / sum(gaps), len(gaps)


def rate_for_eta(cache):
    """(requests per hour, label)."""
    obs = observed_rate(cache)
    if obs is None:
        return FALLBACK_RATE_PER_H, "assumed %d requests/hour (too little history to observe a rate)" % FALLBACK_RATE_PER_H
    rate, n = obs
    return rate * 3600, "observed %.0f requests/hour over the last %d results" % (rate * 3600, n)


def hours(n_requests, per_hour):
    return n_requests / per_hour if per_hour > 0 else float("inf")


def phase_report(cache, universe, names, indexes, with_windows):
    """Print per-index progress; return (per-index dicts, tokenization missing)."""
    total, tok_missing = tokenize_needs(cache, universe, names)
    say("  tokenizations: %d of %d strings still needed" % (tok_missing, total))
    out = []
    for index in indexes:
        r = analyse_index(cache, universe, names, index, with_windows)
        out.append(r)
        line = "  %-24s finds %d done / %d missing" % (index, r["finds_done"], r["finds_missing"])
        if r["undecided_names"]:
            line += " (+extras unknown for %d names)" % r["undecided_names"]
        if with_windows:
            if r["undecided_strings"]:
                line += "; windows %d done / %d missing over decidable strings, UNKNOWN for %d strings (finds missing)" % (
                    r["windows_done"], r["windows_missing"], r["undecided_strings"])
            else:
                line += "; windows %d done / %d missing" % (r["windows_done"], r["windows_missing"])
        say(line)
    return out, tok_missing


# -- commands --------------------------------------------------------------------------
class UsageError(Exception):
    """A problem with how the command was called (exit 1, message only, no traceback)."""


def phase_setup(cache, args, command):
    """(universe, names, indexes, with_windows) for a phase, from the cache."""
    check_shard(cache, args, parse_shard(args.shard))
    prev = cache.get_meta(tkey("task_names", args.task))
    if prev is None:
        raise UsageError("no plan in this cache for task %s: run `plan` first" % args.task)
    all_names = json.loads(prev)
    universe = load_universe(cache, all_names)
    if command == "pilot":
        raw = cache.get_meta(tkey("pilot_names", args.task))
        if raw is None:
            raise UsageError(
                "the pilot list is not set: `plan` picks it once every main name has its OLMo 2 "
                "finds cached (plain and glued; a string whose tokenization failed %d runs counts as plain "
                "only). Rerun `plan` (it resumes) and try again." % TOKENIZE_MAX_FAILS)
        pilot = json.loads(raw)
        gone = [n for n in pilot if n not in universe]
        if gone:
            say("note: %d pilot names are no longer in the task and are skipped: %s" % (len(gone), gone[:5]))
        return universe, [n for n in pilot if n in universe], INDEXES, True
    return universe, list(all_names), INDEXES, True


def cmd_plan(ctx, args):
    cache = ctx.cache
    check_constants(cache, args)
    shard = parse_shard(args.shard)
    check_shard(cache, args, shard)
    names = shard_names(TASKS[args.task](args.prompts), shard)
    with open(args.alias_table, encoding="utf-8") as f:
        alias_table = AliasTable(json.load(f))
    with open(args.snapshot, encoding="utf-8") as f:
        snapshot = json.load(f)
    universe = build_universe(names, alias_table, snapshot)
    added, deactivated, reactivated = sync_entities(cache, universe, args.task)
    cache.set_meta(tkey("task_names", args.task), json.dumps(names, ensure_ascii=False))
    cache.set_meta(tkey("shard", args.task), shard_label(shard))
    record_run_meta(cache, args)
    by_source = {}
    for lst in universe.values():
        for q in lst:
            by_source[q.source] = by_source.get(q.source, 0) + 1
    say("plan: %d names (shard %s), %d query strings %s"
        % (len(names), shard_label(shard), sum(by_source.values()), by_source))
    say("entities: %d added, %d deactivated (rows kept), %d reactivated" % (added, deactivated, reactivated))

    # Tokenize every glued form up front (index-independent, cached), then the
    # OLMo 2 finds for all names through the common pipeline, without windows.
    seen = set()
    for name in names:
        for q in universe[name]:
            if q.string not in seen:
                seen.add(q.string)
                ensure_glued(ctx, q.string)
                ctx.tick("tokenize", name)
    say("tokenized %d strings (%d requests sent so far)" % (len(seen), ctx.client.requests_sent))
    run_pipeline(ctx, universe, names, (OLMO2,), with_windows=False)
    say("OLMo 2 finds: %d requests sent in this run, %d of them for extra forms "
        "(stripped, folded, alias and their glued forms)" % (ctx.client.requests_sent, ctx.extra_sent))
    if ctx.undecided:
        say("%d names have a failed main find; their extra forms are retried on the next run" % len(ctx.undecided))
    choose_pilot_once(cache, universe, names, args.task)


def choose_pilot_once(cache, universe, names, task):
    key = tkey("pilot_names", task)
    stored = cache.get_meta(key)
    if stored is not None:
        say("pilot list already stored (%d names), kept unchanged" % len(json.loads(stored)))
        return
    counts, no_plain, no_glued = {}, 0, 0
    for n in names:
        main = main_of(universe[n])
        cnt = pool_cnt(cache, OLMO2, main.string) if main else None
        if cnt is not None:
            counts[n] = cnt
        elif main is None or cache.get_find(OLMO2, main.string, "plain") is None:
            no_plain += 1
        else:
            no_glued += 1  # plain find cached, the glued half is not yet decidable
    if no_plain or no_glued:
        parts = []
        if no_plain:
            parts.append("%d main names still lack an OLMo 2 find" % no_plain)
        if no_glued:
            parts.append("%d main names have the plain find but not yet the glued one (glued tokenization "
                         "or find failed; a string is used plain only after %d failed tokenizations)"
                         % (no_glued, TOKENIZE_MAX_FAILS))
        say("pilot list NOT chosen yet: %s; rerun `plan` (it resumes)" % "; ".join(parts))
        return
    pilot = choose_pilot(counts, SEED)
    cache.set_meta(key, json.dumps(pilot, ensure_ascii=False))
    say("pilot list chosen once and stored: %d names" % len(pilot))


def record_run_meta(cache, args):
    head, dirty = git_info()
    cache.set_meta("constants", json.dumps(constants_now()))
    cache.set_meta("script_version", SCRIPT_VERSION)
    cache.set_meta("git_head", head)
    cache.set_meta("git_dirty", dirty)
    cache.set_meta("plan_time", now_iso())
    for key, path in zip(INPUTS, (args.prompts, args.alias_table, args.snapshot)):
        cache.set_meta("input_sha256_" + key, sha256_file(path))
        # Absolute, so a later command run from another directory finds the same file.
        cache.set_meta("input_path_" + key, os.path.abspath(str(path)))


def cmd_phase(ctx, args):
    """pilot and full: finds + windows on all five indexes, OLMo 2 first."""
    check_constants(ctx.cache, args)
    universe, names, indexes, with_windows = phase_setup(ctx.cache, args, args.command)
    for line in hash_warnings(ctx.cache, args):
        say(line, err=True)
    say("%s: %d names on %d indexes" % (args.command, len(names), len(indexes)))
    run_pipeline(ctx, universe, names, indexes, with_windows)
    if ctx.undecided:
        say("%d (index, name) pairs had a failed main find; their extras are retried on the next run"
            % len(ctx.undecided))


def hash_warnings(cache, args):
    """Loud lines for each input file whose SHA-256 no longer matches the plan's.

    The file checked is the one `plan` recorded (input_path_<k>), unless the
    command line names that input explicitly: a plan run with non-default paths
    must not warn later just because the default files differ.
    """
    lines = []
    for key in INPUTS:
        recorded = cache.get_meta("input_sha256_" + key)
        if recorded is None:
            continue
        path = getattr(args, key)
        if not getattr(args, key + "_explicit", True):
            path = cache.get_meta("input_path_" + key) or path
        try:
            now = sha256_file(path)
        except OSError:
            lines.append("WARNING: input %s (%s) cannot be read now; the plan was built from a file with sha256 %s"
                         % (key, path, recorded[:12]))
            continue
        if now != recorded:
            lines.append("WARNING: input %s changed since `plan` (%s): plan sha256 %s, now %s. "
                         "Rerun `plan` to pick the change up (nothing cached is refetched)."
                         % (key, path, recorded[:12], now[:12]))
    return lines


def cmd_dry_run(cache, args):
    say("DRY RUN: nothing is sent, nothing is written (%s)" % args.command)
    diff = constants_diff(cache)
    if diff:
        say("WARNING: constants differ from the plan's (%s); a real run refuses unless --allow-constants-change"
            % describe_diff(diff))
    if args.command == "plan":
        shard = parse_shard(args.shard)
        check_shard(cache, args, shard)
        names = shard_names(TASKS[args.task](args.prompts), shard)
        with open(args.alias_table, encoding="utf-8") as f:
            alias_table = AliasTable(json.load(f))
        with open(args.snapshot, encoding="utf-8") as f:
            snapshot = json.load(f)
        universe = build_universe(names, alias_table, snapshot)
        by_source = {}
        for lst in universe.values():
            for q in lst:
                by_source[q.source] = by_source.get(q.source, 0) + 1
        say("%d names (shard %s), %d query strings %s (from the input files)"
            % (len(names), shard_label(shard), sum(by_source.values()), by_source))
        indexes, with_windows = (OLMO2,), False
    else:
        universe, names, indexes, with_windows = phase_setup(cache, args, args.command)
        say("%s: %d names on %d indexes" % (args.command, len(names), len(indexes)))
    reports, tok_missing = phase_report(cache, universe, names, indexes, with_windows)
    known = tok_missing + sum(r["finds_missing"] + r["windows_missing"] for r in reports)
    per_h, label = rate_for_eta(cache)
    say("requests still to send, counting only what the cache lets us derive: %d "
        "(%.1f h at %s)" % (known, hours(known, per_h), label))
    if any(r["undecided_names"] or r["undecided_strings"] for r in reports):
        say("this is a lower bound: extras and windows of undecided names are not counted")
    if with_windows:
        extrapolate(cache, names, indexes, reports)


def extrapolate(cache, names, indexes, reports):
    say("EXTRAPOLATION (not a count):")
    olmo = reports[0]
    if indexes[0] != OLMO2 or not names or olmo["finds_missing"] or olmo["undecided_names"] or olmo["undecided_strings"]:
        say("  unavailable: the OLMo 2 finds of these names are not complete yet (run `plan` first)")
        return
    n = len(names)
    per_name = (olmo["finds_done"] + olmo["windows_total"]) / n
    rest = 0.0
    for idx, r in zip(indexes[1:], reports[1:]):
        rest += max(0.0, per_name * n - (r["finds_done"] + r["windows_done"]))
    per_h, label = rate_for_eta(cache)
    say("  OLMo 2 needs %.1f requests per name (finds + windows). Assuming each of the other %d indexes "
        "needs the same (OLMo 2 is the largest corpus, so likely an overestimate): about %.0f more "
        "requests." % (per_name, len(indexes) - 1, rest))
    say("  = %.1f h at the %.2f s gap (%.0f requests/hour), %.1f h at %s"
        % (hours(rest, 3600 / MIN_GAP_S), MIN_GAP_S, 3600 / MIN_GAP_S, hours(rest, per_h), label))


def cmd_status(cache, args):
    path = Path(args.cache)
    say("cache: %s (%.1f MB)" % (path, path.stat().st_size / 1e6))
    try:
        free = shutil.disk_usage(str(path.parent)).free
        say("free disk on %s: %.1f GB" % (path.parent, free / 1e9))
    except OSError as e:
        say("free disk: unknown (%s)" % e)

    plan_time = cache.get_meta("plan_time")
    if plan_time is None:
        say("NOTE: no plan has been run on this cache (`plan` records the input hashes)")
    else:
        say("plan run at %s; script version %s; git %s (dirty: %s)" % (
            plan_time, cache.get_meta("script_version"), (cache.get_meta("git_head") or "?")[:12],
            cache.get_meta("git_dirty")))
        say("constants: %s" % cache.get_meta("constants"))
        say("task %s, shard %s" % (args.task, cache.get_meta(tkey("shard", args.task)) or "all (not recorded)"))
        diff = constants_diff(cache)
        if diff:
            say("WARNING: constants differ from this script's (%s); plan, pilot and full refuse "
                "unless --allow-constants-change" % describe_diff(diff))
        warnings = hash_warnings(cache, args)
        for line in warnings:
            say(line)
        if not warnings:
            say("input files: sha256 match the plan (%s)" % ", ".join(INPUTS))

    prev = cache.get_meta(tkey("task_names", args.task))
    conn = cache.conn
    n_active = conn.execute("SELECT COUNT(*) FROM entities WHERE active = 1").fetchone()[0]
    n_inactive = conn.execute("SELECT COUNT(*) FROM entities WHERE active = 0").fetchone()[0]
    say("entities: %d active, %d inactive (dropped by a later plan; their cached rows are kept)" % (n_active, n_inactive))
    raw_pilot = cache.get_meta(tkey("pilot_names", args.task))
    say("pilot list: %s" % ("not set yet" if raw_pilot is None else "%d names" % len(json.loads(raw_pilot))))

    per_h, rate_label = rate_for_eta(cache)
    if prev is not None:
        all_names = json.loads(prev)
        universe = load_universe(cache, all_names)
        phases = [("plan", all_names, (OLMO2,), False)]
        if raw_pilot is not None:
            pn = [n for n in json.loads(raw_pilot) if n in universe]
            phases.append(("pilot", pn, INDEXES, True))
        phases.append(("full", all_names, INDEXES, True))
        for label, names, indexes, with_windows in phases:
            say("phase %s: %d names, %d indexes" % (label, len(names), len(indexes)))
            reports, tok_missing = phase_report(cache, universe, names, indexes, with_windows)
            remaining = sum(r["finds_missing"] + r["windows_missing"] for r in reports)
            if label != "pilot":
                remaining += tok_missing
            bound = " (lower bound: extras and windows of undecided names are unknown)" if any(
                r["undecided_names"] or r["undecided_strings"] for r in reports) else ""
            say("  remaining (known so far): %d requests%s, ETA %.1f h at %s" % (
                remaining, bound, hours(remaining, per_h), rate_label))

    say("errors by HTTP status: %s" % (cache.error_counts() or "none"))
    n400 = cache.error_counts().get("400", 0)
    if n400:
        say("  %d HTTP 400 errors: our bug (a malformed request); see the errors table" % n400)
    last = cache.last_error()
    if last:
        say("last error: %s %s %s status=%s %s" % (
            last["ts"], last["index_name"], last["query_type"], last["http_status"], last["body_snippet"][:120]))

    if prev is not None:
        universe = load_universe(cache, json.loads(prev))
        failed = sorted({q.string for lst in universe.values() for q in lst
                         if glued_state(cache, q.string)[0] == "failed"})
        unknown = sum(1 for lst in universe.values() for q in lst if glued_state(cache, q.string)[0] == "unknown")
        say("glued-tokenization failures: %d strings have no glued form %s" % (len(failed), failed[:5]))
        say("glued forms not tokenized yet: %d strings" % unknown)
        strings = {q.string for lst in universe.values() for q in lst}
        counted = {s: tokenize_fails(cache, s) for s in strings}
        abandoned = sorted(s for s, n in counted.items()
                           if n >= TOKENIZE_MAX_FAILS and cache.get_glued_ids("(" + s) is None)
        retrying = sorted(s for s, n in counted.items()
                          if 0 < n < TOKENIZE_MAX_FAILS and cache.get_glued_ids("(" + s) is None)
        say("tokenization given up after %d failed runs (plain form only): %d strings %s"
            % (TOKENIZE_MAX_FAILS, len(abandoned), abandoned[:5]))
        say("tokenization failing, still retried: %d strings %s" % (len(retrying), retrying[:5]))


# -- entry point ---------------------------------------------------------------------------
def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", choices=sorted(TASKS), default=DEFAULT_TASK, help="which benchmark task (default: %(default)s)")
    # Input defaults are filled in by main(), so it can tell "explicit" from "default":
    # status, pilot and full check the files `plan` recorded unless one is given here.
    common.add_argument("--prompts", type=Path, default=None, help="prompts.csv (default: %s)" % DEFAULT_PROMPTS)
    common.add_argument("--alias-table", type=Path, default=None, help="alias_table.json (default: %s)" % DEFAULT_ALIAS_TABLE)
    common.add_argument("--snapshot", type=Path, default=None, help="wikidata_snapshot.json (default: %s)" % DEFAULT_SNAPSHOT)
    common.add_argument("--allow-constants-change", action="store_true",
                        help="accept CAP_FULL/SAMPLE_N/DISP_LEN/SEED/MIN_GAP_S values that differ from the ones "
                             "stored by an earlier plan (plan, pilot and full refuse otherwise)")
    common.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="SQLite cache (default: %(default)s)")
    common.add_argument("--shard", default=None, metavar="I/N",
                        help="fetch only the I-th of N shards of the task's names; one cache per shard")
    common.add_argument("--max-requests", type=int, default=None, metavar="N",
                        help="stop cleanly after N network requests; the cache resumes")
    common.add_argument("--dry-run", action="store_true",
                        help="send nothing and write nothing; print the request counts derivable from the cache")
    ap = argparse.ArgumentParser(
        description="Resumable, rate-limited infini-gram window fetcher. Run plan, then pilot, then full.")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, text in (
        ("plan", "tokenize glued forms, run the OLMo 2 finds, choose the pilot names once"),
        ("pilot", "finds and windows for the pilot names on all five indexes"),
        ("full", "finds and windows for every name on all five indexes, OLMo 2 first"),
        ("status", "progress, errors and ETA from the cache; no network"),
    ):
        sub.add_parser(name, parents=[common], help=text, description=text)
    return ap


def main(argv=None, transport=None, clock=None, sleep=None, wall=None) -> int:
    args = build_parser().parse_args(argv)
    for key in INPUTS:
        setattr(args, key + "_explicit", getattr(args, key) is not None)
        if getattr(args, key) is None:
            setattr(args, key, INPUT_DEFAULTS[key])
    if args.max_requests is not None and args.max_requests < 0:
        say("--max-requests must be >= 0", err=True)
        return 1
    if args.command == "status" or args.dry_run:
        return read_only(args)
    if args.command != "plan" and not Path(args.cache).exists():
        # Do not create an empty cache just to say there is no plan in it.
        say("error: no cache at %s: run `plan` first" % args.cache, err=True)
        return 1

    try:
        cache = Cache(str(args.cache), lock=True)
    except LockHeld as e:
        say(str(e), err=True)
        return 2
    try:
        throttle = Throttle(
            MIN_GAP_S, clock or time.monotonic, sleep or time.sleep,
            last_start_wall=cache.last_request_wall(), wall=wall)
        client = InfinigramClient(
            transport if transport is not None else HttpTransport(), throttle, cache, args.max_requests)
        ctx = Ctx(client, cache)
        try:
            if args.command == "plan":
                cmd_plan(ctx, args)
            else:
                cmd_phase(ctx, args)
        except MaxRequestsReached:
            say("stopped at --max-requests %d: %d requests sent. The cache resumes: rerun the same command."
                % (args.max_requests, client.requests_sent))
            if ctx.failed:
                say("%d items unfetched, rerun to retry" % ctx.failed, err=True)
                return 5
            return 0
        except Blocked:
            say("blocked by infini-gram; wait and rerun, the cache resumes (%d requests sent)" % client.requests_sent, err=True)
            return 3
        except TooManyFailures:
            say("too many consecutive failures; check the error log (status) and rerun, the cache resumes "
                "(%d requests sent)" % client.requests_sent, err=True)
            return 4
        except KeyboardInterrupt:
            say("interrupted (Ctrl+C) after %d requests; the cache resumes: rerun the same command." % client.requests_sent, err=True)
            return 130
        except UsageError as e:
            say("error: %s" % e, err=True)
            return 1
        except Exception as e:  # noqa: BLE001 - last line of defence: report, keep the cache consistent
            say("error: %s: %s" % (type(e).__name__, e), err=True)
            return 1
        if ctx.failed:
            # Never a "done" with exit 0 over failures: a wrapper script would stop retrying.
            say("%d items unfetched, rerun to retry (see `status` and the errors table): %d network requests "
                "sent, %d results served from the cache" % (ctx.failed, client.requests_sent, ctx.hits), err=True)
            return 5
        say("done: %d network requests sent, %d results served from the cache" % (client.requests_sent, ctx.hits))
        return 0
    finally:
        cache.close()  # releases the lock, also on Ctrl+C


def read_only(args):
    """status and --dry-run: no network, no lock, and never create a missing cache."""
    path = Path(args.cache)
    scratch = False
    if not path.exists():
        say("no cache at %s (nothing created)" % path)
        if not (args.command == "plan" and args.dry_run):
            return 0
        # plan --dry-run can still report the query strings and what a fresh cache would need.
        scratch = True
    try:
        cache = Cache(":memory:" if scratch else str(path), lock=False)
    except Exception as e:  # noqa: BLE001
        say("error: cannot open the cache: %s: %s" % (type(e).__name__, e), err=True)
        return 1
    try:
        if args.dry_run:
            cmd_dry_run(cache, args)
        else:
            cmd_status(cache, args)
        return 0
    except UsageError as e:
        say("error: %s" % e, err=True)
        return 1
    except Exception as e:  # noqa: BLE001
        say("error: %s: %s" % (type(e).__name__, e), err=True)
        return 1
    finally:
        cache.close()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # names and tokens are not cp1252
        except AttributeError:
            pass
    sys.exit(main())
