"""Tests for hallutrace/abstention.py. Run: python -m tests.test_abstention"""
from hallutrace.abstention import AbstentionDetector, PHRASES_RELEASED, PHRASES_FIXED, disagreements

fixed = AbstentionDetector("fixed")
released = AbstentionDetector("released")

# The three strings the released detector misses. These are the regression tests.
KNOWN_MISSES = [
    "I am sorry, I cannot help with that.",
    "I cannot answer that question.",
    "I am unable to answer this.",
]

# Abstentions the released detector already catches. The fix must not lose them.
ALREADY_CAUGHT = [
    "I could not find any information about that.",
    "Unfortunately, there is nothing I can offer here.",
    "I apologize, but no sources found.",
    "I'm afraid I do not know.",
]

# Real answers. Neither mode should fire.
REAL_ANSWERS = [
    "Ada Lovelace was born in 1815 in London.",
    "The capital of Thailand is Bangkok.",
    "Title: Attention Is All You Need; Title: BERT;",
]

fails = []

print("released list: %d phrases   fixed list: %d phrases" % (len(PHRASES_RELEASED), len(PHRASES_FIXED)))
print("the bug is visible here ->", [p for p in PHRASES_RELEASED if "sorry" in p.lower()])
print()

print("1. strings the released detector misses")
for s in KNOWN_MISSES:
    r, f = released.is_abstained(s), fixed.is_abstained(s)
    ok = (not r) and f
    print("   %s released=%-5s fixed=%-5s  %r" % ("OK " if ok else "FAIL", r, f, s))
    if not ok:
        fails.append(("known miss", s, r, f))

print("\n2. abstentions the fix must not lose")
for s in ALREADY_CAUGHT:
    r, f = released.is_abstained(s), fixed.is_abstained(s)
    ok = r and f
    print("   %s released=%-5s fixed=%-5s  %r" % ("OK " if ok else "FAIL", r, f, s))
    if not ok:
        fails.append(("regression", s, r, f))

print("\n3. real answers, neither should fire")
for s in REAL_ANSWERS:
    r, f = released.is_abstained(s), fixed.is_abstained(s)
    ok = (not r) and (not f)
    print("   %s released=%-5s fixed=%-5s  %r" % ("OK " if ok else "FAIL", r, f, s))
    if not ok:
        fails.append(("false positive", s, r, f, fixed.matched(s)))

print("\n4. non-string inputs count as abstention, matching the released code")
for v in (None, float("nan"), 0):
    ok = fixed.is_abstained(v) and released.is_abstained(v)
    print("   %s %r" % ("OK " if ok else "FAIL", v))
    if not ok:
        fails.append(("non-string", v))

print("\n5. disagreement counter over the whole sample")
print("   ", disagreements(KNOWN_MISSES + ALREADY_CAUGHT + REAL_ANSWERS))

print()
if fails:
    print("%d FAILURES" % len(fails))
    for f in fails:
        print("  ", f)
    raise SystemExit(1)
print("all tests passed")
