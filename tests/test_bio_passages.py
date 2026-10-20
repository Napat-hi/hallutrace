"""Tests for hallutrace/bio_passages.py (the passages the validator saw). No network."""

import sys

from hallutrace import bio_passages as bp

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r)" % (label, got))


P1 = [{"title": "Ilhan Omar", "text": "<s>She served as campaign manager.</s><s>Minimum wage."}]
P2 = [{"title": "Rin Iwanaga", "text": "Rin Iwanaga is a Japanese actor."}]
c1 = {"Ilhan Omar#Ilhan Omar She was a campaign manager.": P1}
c2 = {"Rin Iwanaga#Rin Iwanaga Rin is an actor.": P2, "Ilhan Omar#Ilhan Omar She was a campaign manager.": P2}

print("1. FActScore's cache key")
eq(bp.cache_keys("Ilhan Omar", "She was a campaign manager."),
   ["Ilhan Omar#Ilhan Omar She was a campaign manager."], "topic#topic atom")
eq(len(bp.cache_keys("A", " x ")), 2, "the atom as given and stripped")

print("2. lookup across caches")
items = [{"item_id": "b-001", "topic": "Ilhan Omar", "atom": "She was a campaign manager."},
         {"item_id": "b-002", "topic": "Rin Iwanaga", "atom": " Rin is an actor. "},
         {"item_id": "b-003", "topic": "Nobody", "atom": "Nothing."}]
got = bp.lookup(iter([c1, c2]), items)
eq(got["b-001"], P1, "found in the first cache; a later cache does not overwrite it")
eq(got["b-002"], P2, "found in another directory's cache, with the atom stripped")
eq(got["b-003"], None, "an atom no cache holds is None, not dropped")

print("3. cleaning")
eq(bp.clean(P1)[0]["text"], "She served as campaign manager. Minimum wage.", "<s> and </s> removed, paragraphs kept apart")
eq(bp.clean(None), None, "None stays None")

print()
if failures:
    print("%d FAILED" % len(failures))
    sys.exit(1)
print("all passed")
