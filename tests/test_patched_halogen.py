"""Checks that the patched HALoGEN tree holds exactly one abstention phrase list.

Skips when external/HALoGEN is not checked out, so it is safe to run anywhere.
Run from the repo root:

    python -m tests.test_patched_halogen
"""

import ast
import os
import sys

from hallutrace.abstention import PHRASES_FIXED

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HALOGEN = os.path.join(ROOT, "external", "HALoGEN")
TARGETS = (
    os.path.join("scorers", "abstention_detector.py"),
    os.path.join("src", "scientific_attribution.py"),
)


def string_lists(path):
    """Every literal list of more than ten strings in a file."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and len(node.elts) > 10:
            if all(
                isinstance(e, ast.Constant) and isinstance(e.value, str)
                for e in node.elts
            ):
                found.append([e.value for e in node.elts])
    return found


def main():
    if not os.path.isdir(HALOGEN):
        print("SKIP: external/HALoGEN is not checked out. See README Setup.")
        return 0

    failures = []
    for rel in TARGETS:
        path = os.path.join(HALOGEN, rel)
        if not os.path.isfile(path):
            failures.append("%s is missing" % rel)
            continue
        lists = string_lists(path)
        if lists:
            failures.append(
                "%s still carries an inline phrase list of %d entries. The patch "
                "should have replaced it with an import from hallutrace.abstention."
                % (rel, len(lists[0]))
            )
        else:
            print("   OK   %s holds no inline phrase list" % rel)

    src = open(os.path.join(HALOGEN, TARGETS[1]), encoding="utf-8").read()
    if "from hallutrace.abstention import" not in src:
        failures.append("src/scientific_attribution.py does not import the shared list")
    else:
        print("   OK   src/scientific_attribution.py imports the shared list")

    # The list the whole pipeline now agrees on.
    print("   OK   one list, %d phrases" % len(PHRASES_FIXED))

    if failures:
        print()
        print("%d FAILED:" % len(failures))
        for f in failures:
            print("  " + f)
        return 1
    print()
    print("all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
