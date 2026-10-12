"""Tests for hallutrace/outcomes.py. No key, no network."""

import sys

from hallutrace.abstention import AbstentionDetector
from hallutrace import outcomes as oc

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


det = AbstentionDetector("fixed")

print("1. the three situations the released code cannot tell apart")
eq(oc.classify("", det), (oc.EMPTY, oc.EMPTY_RESPONSE), "an empty string")
eq(oc.classify("   \n ", det), (oc.EMPTY, oc.EMPTY_RESPONSE), "whitespace only")
eq(oc.classify(None, det), (oc.EMPTY, oc.EMPTY_RESPONSE), "a null response")
eq(oc.classify(float("nan"), det), (oc.EMPTY, oc.EMPTY_RESPONSE),
   "a pandas NaN, which arrives as a float")
eq(oc.classify("I cannot answer that.", det), (oc.ABSTENTION, oc.ABSTAINED),
   "an abstention")
eq(oc.classify("Ada Lovelace was born in 1815.", det), (oc.OK, None),
   "a real answer")

print("2. empty is checked before abstention, and that is the point")
eq(oc.classify("", None)[1], oc.EMPTY_RESPONSE,
   "an empty string is empty even with no detector, because the detector "
   "matches substrings and an empty string contains none of the 30 phrases")
check(det.is_abstained("") is False,
      "the abstention detector alone would call an empty response NOT abstained")

print("3. the reason recorded after decomposition")
eq(oc.reason_after_decomposition(oc.OK, ["a claim"]), None,
   "a decomposed answer stores no reason")
eq(oc.reason_after_decomposition(oc.OK, []), oc.NO_UNITS_FOUND,
   "an answer the decomposer found nothing in is a decomposer result")
eq(oc.reason_after_decomposition(oc.EMPTY, []), oc.EMPTY_RESPONSE,
   "an empty response keeps its own reason")
eq(oc.reason_after_decomposition(oc.ABSTENTION, []), oc.ABSTAINED,
   "an abstention keeps its own reason")

print("4. the trailing clause rule")
kept, dropped = oc.strip_incomplete_tail(
    "Ada was born in 1815. She wrote notes on the engine. She was the daughter of"
)
eq(kept, "Ada was born in 1815. She wrote notes on the engine.",
   "the complete sentences are kept")
eq(dropped, "She was the daughter of",
   "the fragment is returned, not discarded, so the run log can report how "
   "much text the rule removed")

kept, dropped = oc.strip_incomplete_tail("Ada was born in 1815.")
eq(dropped, "", "a response that ends properly loses nothing")
eq(kept, "Ada was born in 1815.", "and is returned unchanged")

kept, dropped = oc.strip_incomplete_tail("She was the daughter of Lord")
eq(kept, "", "a response with no complete sentence keeps nothing")
eq(dropped, "She was the daughter of Lord",
   "and the caller must treat that as no_units_found, not as a score of zero")

kept, dropped = oc.strip_incomplete_tail('He said "yes." Then he left and')
eq(kept, 'He said "yes."', "a closing quote after the full stop still ends "
   "that sentence")

kept, dropped = oc.strip_incomplete_tail("One. Two! Three? Four and")
eq(kept, "One. Two! Three?", "all three terminal marks end a sentence")
eq(dropped, "Four and", "only the fragment goes")

print("   a line break is a boundary too")
LIST = ("The answer is:\n\nSeptember, October, November, December\n\n"
        "The answer is:\n\nSept")
kept, dropped = oc.strip_incomplete_tail(LIST)
eq(kept, "The answer is:\n\nSeptember, October, November, December\n\nThe answer is:",
   "a list answer keeps every finished line, though none ends in a full stop")
eq(dropped, "Sept", "only the unfinished last line goes")
check(oc.ends_mid_clause(LIST), "a list cut mid line is still cut")
check(not oc.ends_mid_clause("Mercury, Mars, Venus\n"),
      "a list whose last line was finished is not cut")
check(not oc.ends_mid_clause("Mercury, Mars, Venus\n  "),
      "trailing spaces after the line break do not change that")
eq(oc.strip_incomplete_tail("Mercury, Mars, Venus\n"), ("Mercury, Mars, Venus\n", ""),
   "and it is returned unchanged")
kept, dropped = oc.strip_incomplete_tail("Mercury was the first.\nThen Mars and")
eq((kept, dropped), ("Mercury was the first.", "Then Mars and"),
   "a sentence end and a line break at the same place agree")
kept, dropped = oc.strip_incomplete_tail("First line\nSecond. Third and")
eq((kept, dropped), ("First line\nSecond.", "Third and"),
   "a full stop after the last line break wins, so a line is cut at its last sentence")
kept, dropped = oc.strip_incomplete_tail("Mercury, Mars, Ven")
eq((kept, dropped), ("", "Mercury, Mars, Ven"),
   "a single unfinished line with no boundary at all still empties")

eq(oc.strip_incomplete_tail(None), ("", ""), "a null response is handled")
eq(oc.strip_incomplete_tail(""), ("", ""), "an empty response is handled")

print("5. batch counts")
batch = [
    {"response": "Ada was born in 1815."},
    {"response": ""},
    {"response": "I cannot answer that."},
    {"response": "Babbage designed the engine."},
]
s = oc.summarise(batch, det)
eq(s["n"], 4, "four records")
eq(s[oc.OK], 2, "two scored")
eq(s[oc.EMPTY], 1, "one empty")
eq(s[oc.ABSTENTION], 1, "one abstention")
eq(round(s["scored_share"], 2), 0.5, "half the batch reaches the decomposer")

print("6. the column takes exactly five values")
eq(oc.REASONS,
   ("empty_response", "abstained", "no_units_found", "failed_parse",
    "failed_api"),
   "the five spellings, in order")
eq(len(set(oc.REASONS)), 5, "and no duplicates")
check(oc.NO_UNITS_FOUND == "no_units_found",
      "the spelling is no_units_found, not ran_no_claims")

print("7. record_decomposition always returns a pair the store accepts")
eq(oc.record_decomposition(oc.OK, ["a claim", "another"]), (2, None),
   "a decomposed answer stores its count and no reason")
eq(oc.record_decomposition(oc.OK, []), (0, oc.NO_UNITS_FOUND),
   "an answer with nothing in it is a decomposer result")
eq(oc.record_decomposition(oc.EMPTY, []), (0, oc.EMPTY_RESPONSE),
   "an empty response keeps its own reason")
eq(oc.record_decomposition(oc.ABSTENTION, []), (0, oc.ABSTAINED),
   "an abstention keeps its own reason")

print("   the two reasons this module cannot see for itself")


class _FakeTimeout(Exception):
    pass


class _RateLimitError(Exception):
    pass


class _Weird(Exception):
    pass


eq(oc.record_decomposition(oc.OK, None, error=_FakeTimeout()),
   (0, oc.FAILED_API), "a timeout is an API failure, worth retrying")
eq(oc.record_decomposition(oc.OK, None, error=_RateLimitError()),
   (0, oc.FAILED_API), "so is a rate limit")
eq(oc.record_decomposition(oc.OK, None, error=ConnectionError()),
   (0, oc.FAILED_API), "and so is anything that is an OSError underneath")
eq(oc.record_decomposition(oc.OK, None, error=_Weird()),
   (0, oc.FAILED_PARSE),
   "anything else is a parse failure, which is not worth retrying")
eq(oc.record_decomposition(oc.OK, None, error=ValueError("bad json")),
   (0, oc.FAILED_PARSE), "a ValueError from json is a parse failure")


class _HasStatus(Exception):
    status_code = 503


eq(oc.record_decomposition(oc.OK, None, error=_HasStatus()),
   (0, oc.FAILED_API),
   "an exception carrying an HTTP status is classified by the status, so a "
   "provider whose class name says nothing is still caught")

check(oc._looks_like_api_error(_FakeTimeout()) is True,
      "classification is by exception type, never by message text, since the "
      "wording differs across the three providers we verify through")

print("   a caller that contradicts itself is refused, not silently stored")
raised = False
try:
    oc.record_decomposition(oc.EMPTY, ["a claim"])
except ValueError as e:
    raised = "would reject the row" in str(e)
check(raised, "outcome EMPTY with units produced raises rather than writing a "
              "row the store would refuse")

print("8. row_is_valid mirrors the store's two CHECK constraints")
check(oc.row_is_valid(None, None), "undecomposed: no count, no reason")
check(not oc.row_is_valid(None, oc.ABSTAINED), "no count but a reason is invalid")
check(oc.row_is_valid(0, oc.EMPTY_RESPONSE), "zero units must carry a reason")
check(not oc.row_is_valid(0, None), "zero units with no reason is invalid")
check(oc.row_is_valid(3, None), "units found means no reason")
check(not oc.row_is_valid(3, oc.NO_UNITS_FOUND), "units found with a reason is invalid")
check(not oc.row_is_valid(0, "ran_no_claims"),
      "the spec's spelling is not one of the five and is refused")
check(not oc.row_is_valid(-1, None), "a negative count is invalid")

print("9. and the mirror is checked against the real store, not asserted")
try:
    import sqlite3
    import tempfile

    from hallutrace import store

    tmp = tempfile.mkdtemp(prefix="hallutrace_outcomes_")
    conn = store.connect(tmp + "/t.sqlite")
    store.init_db(conn)
    conn.execute(
        "INSERT INTO prompts (prompt_id, task, source_file, source_row, prompt,"
        " task_meta) VALUES ('biographies-00001', 'biographies', 'f.csv', 1,"
        " 'Tell me a bio.', '{}')"
    )
    conn.commit()

    seq = [0]

    def store_accepts(n_units, reason):
        seq[0] += 1
        try:
            conn.execute(
                "INSERT INTO generations (generation_id, prompt_id, model,"
                " sample_index, temperature, top_p, seed, escalated, response,"
                " abstained, n_units, decomposer_empty_reason, generated_at,"
                " gen_config_hash, batch_id, finish_reason, n_output_tokens,"
                " max_new_tokens, n_prompt_tokens, engine, gpu, model_revision)"
                " VALUES (?, 'biographies-00001', 'olmo2-7b', 0, 0.0, 1.0, 1, 0,"
                " 'r', 0, ?, ?, '2026-09-24T00:00:00Z', 'abc123def456', 'b0',"
                " 'stop', 12, 128, 20, 'vllm-0.11.0', 'H100', 'deadbeef')",
                ("olmo2-7b|biographies-00001|%d" % seq[0], n_units, reason),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            conn.rollback()
            return False

    grid = []
    for n in (None, 0, 1, 3):
        for r in (None,) + oc.REASONS + ("ran_no_claims", "nonsense"):
            grid.append((n, r))

    # nothing row_is_valid accepts may be rejected by the store
    we_accept_store_rejects = [
        (n, r) for n, r in grid if oc.row_is_valid(n, r) and not store_accepts(n, r)
    ]
    eq(we_accept_store_rejects, [],
       "every row row_is_valid accepts, the store accepts too, over all %d "
       "combinations" % len(grid))

    # and nothing the store accepts is refused by row_is_valid
    store_accepts_we_reject = sorted(
        (str(n), r) for n, r in grid
        if store_accepts(n, r) and not oc.row_is_valid(n, r)
    )
    eq(store_accepts_we_reject, [],
       "the store no longer accepts anything row_is_valid would refuse, over "
       "all %d combinations" % len(grid))
    check(not oc.row_is_valid(None, oc.ABSTAINED),
          "row_is_valid refuses that shape, and always did")

    # With n_units NULL the first CHECK evaluates to NULL, which SQLite treats
    # as satisfied; a second CHECK (n_units IS NOT NULL OR
    # decomposer_empty_reason IS NULL) closes that gap.
    eq(conn.execute(
        "SELECT (NULL IS NULL AND 'abstained' IS NULL)"
        " OR (NULL = 0 AND 'abstained' IS NOT NULL)"
        " OR (NULL > 0 AND 'abstained' IS NULL)").fetchone()[0],
       None,
       "the original constraint by itself still evaluates to NULL for this "
       "shape -- the fix is an added CHECK, not a rewrite of this one")
    check(not store_accepts(None, oc.ABSTAINED),
          "and the store rejects it now: the second CHECK catches what the "
          "first one's three-valued logic let through")

    conn.close()
except ImportError:
    print("   SKIP  hallutrace.store not importable here")

print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("all tests passed")
