r"""Run every tests/test_*.py module and report pass/fail.

Also checks that no test creates or modifies db/api_cache.sqlite, the real API
cache, which cannot be regenerated. Tests must point HALLUTRACE_CACHE_PATH at a
temporary file.

Usage (from the repo root): python scripts/run_tests.py
"""

import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hallutrace import cache  # noqa: E402

PROD_CACHE = os.path.join(ROOT, cache.DEFAULT_CACHE_PATH)


def cache_state():
    """(exists, mtime_ns, size) for the production cache path, or (False, None, None)."""
    if not os.path.exists(PROD_CACHE):
        return (False, None, None)
    st = os.stat(PROD_CACHE)
    return (True, st.st_mtime_ns, st.st_size)


def discover_modules():
    paths = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    return ["tests." + os.path.splitext(os.path.basename(p))[0] for p in paths]


def main() -> int:
    modules = discover_modules()
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT
    env["PYTHONIOENCODING"] = "utf-8"

    baseline = cache_state()
    failed, polluted_by = [], []
    for mod in modules:
        proc = subprocess.run(
            [sys.executable, "-m", mod], cwd=ROOT, env=env,
            capture_output=True, text=True,
        )
        ok = proc.returncode == 0
        print(("   OK  " if ok else "   FAIL") + " " + mod)
        if not ok:
            tail = (proc.stdout + proc.stderr).splitlines()[-15:]
            failed.append((mod, "\n".join(tail)))

        after = cache_state()
        if after != baseline:
            polluted_by.append(mod)
            baseline = after  # so a second offender is reported too

    print()
    if polluted_by:
        print(
            "GUARD FAILED: %s was created or modified while running: %s"
            % (PROD_CACHE, ", ".join(polluted_by))
        )
        print(
            "This file cannot be regenerated (D-013) and .gitignore lets it be "
            "`git add`ed (D-021); no test may touch the real path. Point "
            "HALLUTRACE_CACHE_PATH, or whatever the test's own equivalent is, at a "
            "path under a temp directory instead."
        )
        failed.append(("db/api_cache.sqlite guard", "see message above"))

    if failed:
        print("\n%d of %d failed:" % (len(failed), len(modules)))
        for mod, tail in failed:
            print("\n=== %s ===" % mod)
            print(tail)
        return 1

    print("all %d test modules passed, and the production cache was never touched" % len(modules))
    return 0


if __name__ == "__main__":
    sys.exit(main())
