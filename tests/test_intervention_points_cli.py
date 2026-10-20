"""Tests for scripts/intervention_points.py, the labels CLI: labels.csv, labels_sweep.csv,
labels_bio_tau.csv, t5.csv, summary.md and meta.json from one offline run, and its refusals.

One synthetic fixture set over three models, built in a temp dir: claims (results/bio_year_claims.csv
shape), merged count rows (the counting cell), historical_events' tau rows (headline_406 at 5 to 20, plus a
decoy variant), Biographies' tau rows (no variant column, decoys at 5 and 20), a synthetic verifier output copy
(one generation dropped by `stability.ok_claims`) and a phrase cache filled directly. Expected values
are derived by hand from the fixtures and the rules (slot_types.classify, intervention_points rule v3); the
slot list below says what each slot is for.

  tau        headline_406: OLMo 2 2 at 5 to 10, 4 at 11 to 20; DCLM 1, OLMo 1.7 1, RedPajama 2,
             Pythia 3 everywhere. Biographies' own tau (10th): OLMo 2 3, DCLM 3, OLMo 1.7 1,
             RedPajama 2, Pythia 3.
  D1 dclm 00201 birth   S1 never wrong, c zero: typed FABRICATION, no label (the fabrication_s1 line)
  D2 dclm 00202 birth   S1 1 right 1 wrong, n_c 2: RECALL_FAILURE, LONG_TAIL at Biographies' tau
                        (T5 change, no F5 change), High
  D3 dclm 00203 birth   single_mention (its right sample abstained: dropped by ok_claims), FABRICATION,
                        no point, blank confidence
  D4 dclm 00204 birth   S2 BOTH_IN_DATA rho 0.5: at training data, Medium (s2_rho_near_half)
  D5 dclm 00205 birth   S2 n_c 2: undetermined, High; at Biographies' tau LONG_TAIL at training
                        process, Low, coverage_downgrade 2, and sweep_point there (the headline sweep
                        set plus Biographies' own point)
  O1 olmo 00101 birth   S2 n_c 3: RECALL_FAILURE undetermined at 5 to 10, LONG_TAIL at training
                        process at 11 to 20 and at Biographies' tau; sweep_point everywhere
  O2 olmo 00102 birth   S1 never wrong, n_c 3: its type flips like O1's (t5_all, not t5)
  O3 olmo 00103 birth   S3 c 0.4, c unknown, h 3: the only phrase target, hit on phrase 2: phrase_hit,
                        undetermined, High
  O4 olmo 00104 birth   S2 with no h row: no count row, blank confidence (the 7(d) line)
  R2 rpj  00302 birth   S3 c 0.6, BOTH_IN_DATA rho < 0.5: undetermined, Medium (s3_c_near_cut)
  R1 rpj  00665 death   S1 (the s2_rho_near_half slot under the earlier sample rule), c unknown, h zero: untyped,
                        at inference, Medium (no_count_above_0, the S1 line)

No network: scripts.slot_types.HttpTransport and infinigram.HttpTransport are replaced for the
whole module by a class that fails if constructed. The CLI's default --verifier-root and --out point at
temp paths for the module, and every run passes every input path, so no run reads a real verifier output copy,
the real phrase cache or writes into results/. Everything is written in temp dirs, removed at the end.
Run: python -m tests.test_intervention_points_cli
"""

import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from hallutrace import intervention_points as c2
from hallutrace import slot_types as cu
from hallutrace import infinigram
from scripts import intervention_points as c2cli
from scripts import slot_types as cli

failures = []


def check(cond, label):
    print(("   OK  " if cond else "   FAIL") + " " + label)
    if not cond:
        failures.append(label)


def eq(got, want, label):
    check(got == want, label if got == want else "%s  (got %r, want %r)" % (label, got, want))


class NoNetwork:
    """Stands in for HttpTransport for the whole module: constructing it fails loudly."""

    def __init__(self, *a, **k):
        raise AssertionError("HttpTransport constructed in a test")


# ---------------------------------------------------------------------------
# Fixture builders (copied from tests/test_slot_types.py, not imported: importing it runs it)
# ---------------------------------------------------------------------------

CLAIM_COLUMNS = ("generation_id", "model", "sample_index", "unit_id", "name", "kind", "label", "atom",
                 "stated_years", "own_qid", "wikidata_years", "wikidata_match", "namesake_qids",
                 "namesake_year_qids", "conflation_candidate")


def claim(model, pid, sample, stated, own, kind="birth", name="Ada Test"):
    """One row of results/bio_year_claims.csv, all strings."""
    r = {c: "" for c in CLAIM_COLUMNS}
    right = set(stated.split()) <= set(own.split())
    r.update(generation_id="%s|%s|%d" % (model, pid, sample), model=model, sample_index=str(sample),
             unit_id="%s-b%05d" % (model, sample), name=name, kind=kind,
             label="supported" if right else "hallucinated", atom="%s was born in %s." % (name, stated),
             stated_years=stated, own_qid="Q1", wikidata_years=own, wikidata_match=str(right),
             conflation_candidate="False")
    return r


def samples(model, pid, stated_list, own, kind="birth", name="Ada Test", start=1):
    """One claim per sample, samples numbered from `start`."""
    return [claim(model, pid, start + i, s, own, kind=kind, name=name) for i, s in enumerate(stated_list)]


OLMO, DCLM, DOLMA, RPJ, PILE = (cu.MODEL_INDEX[m] for m in
                                ("olmo2-7b", "dclm-7b", "olmo17-7b", "redpajama-7b", "pythia-69b"))


def row(role, year, status, n_c, name, index, fact="birth"):
    """One merged/counts.csv row (strings); index given every time (a wrong index or fact silently
    makes the slot no_count_row). per_string_json holds the plain name as the `name` string."""
    ps = [{"q": name + " alias", "src": "alias", "pool": 10, "sel": "all", "planned": 10, "fetched": 10,
           "matched": 0},
          {"q": name, "src": "name", "pool": 500, "sel": "sample", "planned": 50, "fetched": 50, "matched": 0}]
    return {"role": role, "prompt_name": name, "index": index, "fact": fact, "year": str(year),
            "status": status, "n_c": n_c, "n_c_unit": "docs", "per_string_json": json.dumps(ps),
            "half_width": "full", "matcher": "near_or_cue", "strings": "no_stripped"}


def write_csv(path, columns, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
        return list(r.fieldnames or []), rows


VERIFIER_META = ["generation_id", "prompt_id", "model", "sample_index", "task", "batch_id", "outcome",
              "decomposer_empty_reason", "sent", "input_row", "stripped_tail"]


def write_verifier(root, claims, outcomes=None):
    """A synthetic local copy of the verifier files for `claims` (tests/test_slot_types.py's
    write_verifier without strays): per model, unit <model>-b00000 with a sidecar row per generation
    (outcome "ok" unless `outcomes` says otherwise), inputs and outputs_filtered rows per sent one."""
    outcomes = outcomes or {}
    gens = {c["generation_id"]: c["model"] for c in claims}
    by_model = defaultdict(list)
    for gid in sorted(gens):
        by_model[gens[gid]].append(gid)
    for model, gids in by_model.items():
        (root / "inputs" / model).mkdir(parents=True, exist_ok=True)
        (root / "outputs_filtered" / model).mkdir(parents=True, exist_ok=True)
        metas, names = [], []
        for gid in gids:
            _, pid, sample = gid.split("|")
            outcome = outcomes.get(gid, "ok")
            ok = outcome == "ok"
            metas.append({"generation_id": gid, "prompt_id": pid, "model": model, "sample_index": sample,
                          "task": "biographies", "batch_id": "b", "outcome": outcome,
                          "decomposer_empty_reason": "", "sent": int(ok), "input_row": len(names) if ok else -1,
                          "stripped_tail": ""})
            if ok:
                names.append(gid)
        unit = "%s-b00000" % model
        write_csv(root / "inputs" / model / (unit + ".meta.csv"), VERIFIER_META, metas)
        write_csv(root / "inputs" / model / (unit + ".csv"), ["name", "response", "generation_id"],
                  [{"name": n, "response": "r", "generation_id": n} for n in names])
        write_csv(root / "outputs_filtered" / model / (unit + ".csv"), ["name", "atoms"],
                  [{"name": n, "atoms": "| A.\tTrue |\n"} for n in names])


def fill_cache(path, answers):
    """{(index, phrase): count} straight into a phrase cache (infinigram.Cache, no lock)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    c = infinigram.Cache(str(path), lock=False)
    try:
        c.ensure_counts_table()
        for (index, phrase), n in sorted(answers.items()):
            c.put_count(index, phrase, n, False, 1.0)
    finally:
        c.close()


def blob(path):
    """git hash-object by hand: .gitattributes has `*.csv text eol=lf`, so git hashes a CSV with LF
    line ends (csv.writer writes CRLF)."""
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# ---------------------------------------------------------------------------
# The fixture set
# ---------------------------------------------------------------------------

D, O, R = "dclm-7b", "olmo2-7b", "redpajama-7b"
D3_ABSTAINED = "%s|biographies-00203|2" % D
CLAIMS = (
    samples(D, "biographies-00201", ["1840", "1840"], "1840", name="Eve Fab")                          # D1
    + samples(D, "biographies-00202", ["1850", "1851"], "1850", name="Fay Bioflip")                    # D2
    + samples(D, "biographies-00203", ["1861", "1860"], "1860", name="Gus Single")                     # D3 (+ dropped)
    + samples(D, "biographies-00204", ["1871"] * 5, "1870", name="Hal Both")                           # D4
    + samples(D, "biographies-00205", ["1891"] * 5, "1890", name="Kit Bioonly")                        # D5
    + samples(O, "biographies-00101", ["1802"] * 5, "1801", name="Al Flip")                            # O1
    + samples(O, "biographies-00102", ["1810", "1810"], "1810", name="Bea Never")                      # O2
    + samples(O, "biographies-00103", ["1821", "1821", "1822", "1823", "1824"], "1820", name="Cal Target")  # O3
    + samples(O, "biographies-00104", ["1831"] * 5, "1830", name="Dee Norow")                          # O4
    + samples(R, "biographies-00302", ["1881", "1881", "1881", "1882", "1883"], "1880", name="Jo Near")  # R2
    + samples(R, "biographies-00665", ["1900", "1900", "1905"], "1900", kind="death", name="Ida Rho")  # R1
)
COUNTS = [
    row("c", 1840, "zero", "0", "Eve Fab", DCLM),
    row("c", 1850, "counted", "2", "Fay Bioflip", DCLM), row("h", 1851, "zero", "0", "Fay Bioflip", DCLM),
    row("c", 1860, "zero", "0", "Gus Single", DCLM), row("h", 1861, "zero", "0", "Gus Single", DCLM),
    row("c", 1870, "counted", "4", "Hal Both", DCLM), row("h", 1871, "counted", "4", "Hal Both", DCLM),
    row("c", 1890, "counted", "2", "Kit Bioonly", DCLM), row("h", 1891, "zero", "0", "Kit Bioonly", DCLM),
    row("c", 1801, "counted", "3", "Al Flip", OLMO), row("h", 1802, "zero", "0", "Al Flip", OLMO),
    row("c", 1810, "counted", "3", "Bea Never", OLMO),
    row("c", 1820, "unknown", "", "Cal Target", OLMO), row("h", 1821, "counted", "3", "Cal Target", OLMO),
    row("c", 1830, "counted", "5", "Dee Norow", OLMO),   # no h row for 1831
    row("c", 1880, "counted", "5", "Jo Near", RPJ), row("h", 1881, "counted", "1", "Jo Near", RPJ),
    row("c", 1900, "unknown", "", "Ida Rho", RPJ, fact="death"), row("h", 1905, "zero", "0", "Ida Rho", RPJ,
                                                                     fact="death"),
]
SWEEP = tuple(range(5, 21))
HE_TAU = {p: {OLMO: "2" if p <= 10 else "4", DCLM: "1", DOLMA: "1", RPJ: "2", PILE: "3"} for p in SWEEP}
BIO_TAU = {OLMO: "3", DCLM: "3", DOLMA: "1", RPJ: "2", PILE: "3"}
HE_COLUMNS = ["variant", "index", "model", "percentile", "tau"]
BIO_COLUMNS = ["index", "model", "percentile", "tau"]
EXPECTED_TAU = "%s=2,%s=1,%s=1,%s=2,%s=3" % (OLMO, DCLM, DOLMA, RPJ, PILE)
EXPECTED_BIO_TAU = "%s=3,%s=3,%s=1,%s=2,%s=3" % (OLMO, DCLM, DOLMA, RPJ, PILE)
# The only phrase target: OLMo 2, "Cal Target", 1820 (O3's unknown c row). Phrase 1 is 0, phrase 2 hits.
TARGET_PHRASES = cu.phrases("Cal Target", "birth", 1820)
CACHE_ANSWERS = {(OLMO, TARGET_PHRASES[0]): 0, (OLMO, TARGET_PHRASES[1]): 2}


def he_rows(drop=lambda p, i: False):
    rows = [{"variant": "headline_406", "index": i, "model": i, "percentile": str(p), "tau": t}
            for p in SWEEP for i, t in HE_TAU[p].items() if not drop(p, i)]
    # a decoy variant the CLI must skip
    rows += [{"variant": "with_short_444", "index": i, "model": i, "percentile": str(p), "tau": "9"}
             for p in SWEEP for i in HE_TAU[p]]
    return rows


def bio_rows():
    rows = [{"index": i, "model": i, "percentile": "10", "tau": t} for i, t in BIO_TAU.items()]
    # decoy percentiles the CLI must skip
    rows += [{"index": i, "model": i, "percentile": p, "tau": "7"} for p in ("5", "20") for i in BIO_TAU]
    return rows


# ---------------------------------------------------------------------------
# Expected label rows, by hand (columns as CSV strings)
# ---------------------------------------------------------------------------

RV = c2.RULE_VERSION
UNT = cu.UNTYPED_LABEL
INF, TD, TP, UND = cu.INFERENCE, cu.DATA, cu.PROCESS, cu.UNDETERMINED
BH, BL, FA, LT, RF = cu.BOTH_HIGH, cu.BOTH_LOW, cu.FABRICATION, cu.LONG_TAIL, cu.RECALL_FAILURE


def lab(model, pid, kind, dom, t5, regime, rate, point, conf, tau, n_wrong, cls="typed", n_right="0",
        c="", n_c="", n_h="", rho="", reasons="", downgrade="0", no_count_row="0"):
    return {"prompt_id": pid, "model": model, "claim_id": "%s|%s|%s" % (pid, kind, dom), "phase2_type": t5,
            "regime": regime, "claim_rate": rate, "intervention_point": point, "confidence": conf,
            "coverage_downgrade": downgrade, "rule_version": RV, "kind": kind, "tau": tau, "class": cls,
            "t5_column": t5, "merged": "0", "untyped_reason": UNT if cls in ("phrase_hit", "untyped") else "",
            "open_band": "", "flag": "", "no_count_row": no_count_row, "residual": "0", "low_n": "0",
            "n_right": n_right, "n_wrong": n_wrong, "c": c, "dominant_wrong": dom, "n_c": n_c, "n_h": n_h,
            "rho": rho, "f5_set": point, "confidence_reasons": reasons}


HEAD = [
    lab(D, "biographies-00202", "birth", "1851", RF, "S1", "0.5", INF, "High", "1", "1", n_right="1",
        n_c="2", n_h="0", rho="0"),
    lab(D, "biographies-00203", "birth", "1861", FA, "single_mention", "1", "", "", "1", "1", n_c="0", n_h="0"),
    lab(D, "biographies-00204", "birth", "1871", BH, "S2", "1", TD, "Medium", "1", "5", c="1", n_c="4", n_h="4",
        rho="0.5", reasons="s2_rho_near_half"),
    lab(D, "biographies-00205", "birth", "1891", RF, "S2", "1", UND, "High", "1", "5", c="1", n_c="2", n_h="0",
        rho="0"),
    lab(O, "biographies-00101", "birth", "1802", RF, "S2", "1", UND, "Medium", "2", "5", c="1", n_c="3", n_h="0",
        rho="0", reasons="sweep_point"),
    lab(O, "biographies-00103", "birth", "1821", UNT, "S3", "0.4", UND, "High", "2", "5", cls="phrase_hit",
        c="0.4", n_h="3"),
    lab(O, "biographies-00104", "birth", "1831", "", "S2", "1", "", "", "2", "5", cls="no_count_row", c="1",
        n_c="5", no_count_row="1"),
    lab(R, "biographies-00302", "birth", "1881", BL, "S3", "0.6", UND, "Medium", "2", "5", c="0.6", n_c="5",
        n_h="1", rho="0.1667", reasons="s3_c_near_cut"),
    lab(R, "biographies-00665", "death", "1905", UNT, "S1", "0.3333", INF, "Medium", "2", "1", cls="untyped",
        n_right="2", n_h="0", reasons="no_count_above_0"),
]
KEYS = [(r["model"], r["prompt_id"], r["kind"]) for r in HEAD]
O1K, O3K, O4K = (O, "biographies-00101", "birth"), (O, "biographies-00103", "birth"), (O, "biographies-00104", "birth")
D2K, D5K = (D, "biographies-00202", "birth"), (D, "biographies-00205", "birth")
O2K = (O, "biographies-00102", "birth")
ABSENCE = {"phase2_type": LT, "t5_column": LT, "intervention_point": TP, "f5_set": TP, "confidence": "Low",
           "confidence_reasons": "absence_point; sweep_point"}


def expected_at(p):
    """labels_sweep.csv rows at percentile p (strings, percentile column first)."""
    out = []
    for k, r in zip(KEYS, HEAD):
        r = dict(r)
        if p >= 11 and k[0] == O:
            r["tau"] = "4"
            if k == O1K:
                r.update(ABSENCE, coverage_downgrade="1")
        out.append(dict({"percentile": str(p)}, **r))
    return out


def expected_bio():
    out = []
    for k, r in zip(KEYS, HEAD):
        r = dict(r)
        r["tau"] = {O: "3", D: "3", R: "2"}[k[0]]
        if k == D2K:
            r.update(phase2_type=LT, t5_column=LT)
        elif k == D5K:
            r.update(ABSENCE, coverage_downgrade="2")
        elif k == O1K:
            r.update(ABSENCE, coverage_downgrade="1")
        out.append(r)
    return out


def diff_cols(a, b):
    return sorted(c for c in a if a[c] != b[c])


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


class Env:
    def __init__(self, base):
        self.d = Path(base)
        self.claims, self.counts = self.d / "in" / "claims.csv", self.d / "in" / "counts.csv"
        self.tau, self.bio_tau = self.d / "in" / "he_tau.csv", self.d / "in" / "bio_tau.csv"
        self.verifier = self.d / "verifier"
        self.cache = self.d / "pc" / "phrase_counts.sqlite"
        self.out = self.d / "out"
        write_csv(self.claims, list(CLAIM_COLUMNS), CLAIMS)
        write_csv(self.counts, list(COUNTS[0]), COUNTS)
        write_csv(self.tau, HE_COLUMNS, he_rows())
        write_csv(self.bio_tau, BIO_COLUMNS, bio_rows())
        write_verifier(self.verifier, CLAIMS, outcomes={D3_ABSTAINED: "abstention"})
        fill_cache(self.cache, CACHE_ANSWERS)

    def args(self, *extra, **over):
        a = {"--claims": self.claims, "--counts": self.counts, "--tau": self.tau, "--bio-tau": self.bio_tau,
             "--cache": self.cache, "--verifier-root": self.verifier, "--out": self.out,
             "--expected-tau": EXPECTED_TAU, "--expected-bio-tau": EXPECTED_BIO_TAU}
        for k, v in over.items():
            a["--" + k.replace("_", "-")] = v
        argv = []
        for k, v in a.items():
            if v is not None:
                argv += [k, str(v)]
        return argv + list(extra)


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = c2cli.main(argv)
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


def listing(d):
    d = Path(d)
    return sorted(os.listdir(str(d))) if d.exists() else None


def snapshot(d):
    d = Path(d)
    return {n: (d / n).read_bytes() for n in sorted(os.listdir(str(d)))} if d.exists() else None


# ---------------------------------------------------------------------------
# Module-wide guards
# ---------------------------------------------------------------------------

tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
old = {"http": (cli.HttpTransport, infinigram.HttpTransport), "verifier": cli.DEFAULT_VERIFIER_ROOT,
       "out": c2cli.DEFAULT_OUT, "lock": os.environ.get(infinigram.LOCK_DIR_ENV)}
cli.HttpTransport = infinigram.HttpTransport = NoNetwork
cli.DEFAULT_VERIFIER_ROOT = Path(tmp.name) / "no_default_verifier"
c2cli.DEFAULT_OUT = Path(tmp.name) / "no_default_out"
os.environ[infinigram.LOCK_DIR_ENV] = str(Path(tmp.name) / "userlock")

try:
    env = Env(Path(tmp.name) / "e")
    cache_before, cache_listing = env.cache.read_bytes(), listing(env.cache.parent)

    print("1. the run")
    code, out, err = run(env.args())
    eq((code, err), (0, ""), "exit 0, nothing on stderr")
    eq(listing(env.out), sorted(c2cli.OUTPUTS), "the seven outputs, no temporary file left")
    eq(out.strip(), "11 slots, 9 labelled, 2 never wrong; F5 over 7 (0 open); sweep 5%% to 20%%: 2 quantity(ies) "
       "with a reversal; Biographies' tau: 3 T5 and 2 F5 changes -> %s" % env.out, "the closing line on stdout")
    eq((env.cache.read_bytes() == cache_before, listing(env.cache.parent)), (True, cache_listing),
       "the phrase cache is byte-identical, nothing beside it (no -wal, -shm or .lock)")
    eq(listing(Path(tmp.name) / "userlock"), None, "no per-user infini-gram lock taken")
    first = snapshot(env.out)

    print("2. labels.csv (headline: historical_events, headline_406, 10th percentile)")
    header, rows = read_csv(env.out / c2cli.LABELS_NAME)
    eq(header, list(c2.LABEL_COLUMNS), "header is intervention_points.LABEL_COLUMNS")
    eq(len(rows), 9, "one row per labelled slot: 11 slots less the 2 never wrong")
    eq([(r["model"], r["prompt_id"], r["kind"]) for r in rows], KEYS, "rows in (model, prompt_id, kind) order")
    check(O2K not in KEYS and (D, "biographies-00201", "birth") not in KEYS
          and all((r["model"], r["prompt_id"]) not in ((O, "biographies-00102"), (D, "biographies-00201"))
                  for r in rows), "the never-wrong slots (O2, D1) have no label")
    eq({r["rule_version"] for r in rows}, {RV}, "rule_version on every row")
    for got, want in zip(rows, HEAD):
        eq(got, want, "label %s %s: every column as rule v3 gives it" % (want["model"], want["prompt_id"]))
    by = {(r["model"], r["prompt_id"], r["kind"]): r for r in rows}
    eq((by[(D, "biographies-00203", "birth")]["regime"], by[(D, "biographies-00203", "birth")]["confidence"]),
       ("single_mention", ""), "D3: single_mention after ok_claims drops its abstained right sample; blank confidence")
    eq(sum(1 for r in rows if r["flag"] == c2.FLAG_FABRICATION_S1), 0, "fabrication_s1: 0 labelled slots")

    print("3. labels_sweep.csv (headline_406 at every percentile 5 to 20)")
    header, srows = read_csv(env.out / c2cli.SWEEP_NAME)
    # slimmed: the key, the percentile and tau, and what can move with tau
    eq(c2cli.SWEEP_COLUMNS, ["prompt_id", "model", "kind", "percentile", "tau", "phase2_type", "t5_column",
                             "intervention_point", "confidence", "confidence_reasons"], "SWEEP_COLUMNS, the ten")
    eq(header, c2cli.SWEEP_COLUMNS, "header: SWEEP_COLUMNS")
    smeta = json.loads((env.out / c2cli.META_NAME).read_text(encoding="utf-8"))
    eq((smeta["sweep_columns"], smeta["sweep_dropped_columns"]),
       (c2cli.SWEEP_COLUMNS, [c for c in c2.LABEL_COLUMNS if c not in c2cli.SWEEP_COLUMNS]),
       "meta.json names the sweep file's columns and the dropped ones, in LABEL_COLUMNS order")
    eq(len(srows), 16 * 9, "16 percentiles x 9 labelled slots")
    byp = defaultdict(list)
    for r in srows:
        byp[r["percentile"]].append(r)
    eq(sorted(byp, key=int), [str(p) for p in SWEEP], "percentiles 5 to 20")
    eq({p: [(r["model"], r["prompt_id"], r["kind"]) for r in v] for p, v in byp.items()},
       {str(p): KEYS for p in SWEEP}, "the same slots, in the same order, at every percentile")
    for p in SWEEP:
        eq(byp[str(p)], [{c: x[c] for c in c2cli.SWEEP_COLUMNS} for x in expected_at(p)],
           "percentile %d: every row" % p)
    eq({p: sorted({(r["model"], r["tau"]) for r in byp[str(p)]}) for p in (5, 10, 11, 20)},
       {5: [(D, "1"), (O, "2"), (R, "2")], 10: [(D, "1"), (O, "2"), (R, "2")],
        11: [(D, "1"), (O, "4"), (R, "2")], 20: [(D, "1"), (O, "4"), (R, "2")]},
       "tau column per model and percentile (OLMo 2 2 to 10, 4 from 11)")
    head_by = {k: r for k, r in zip(KEYS, rows)}
    sc = [c for c in c2cli.SWEEP_COLUMNS if c != "percentile"]
    for p in SWEEP:
        changed = {(r["model"], r["prompt_id"], r["kind"]): diff_cols(
            {c: head_by[(r["model"], r["prompt_id"], r["kind"])][c] for c in sc}, {c: r[c] for c in sc})
                   for r in byp[str(p)]}
        changed = {k: [c for c in v if c != "tau"] for k, v in changed.items()}
        want = {k: [] for k in KEYS}
        if p >= 11:
            want[O1K] = ["confidence", "confidence_reasons", "intervention_point", "phase2_type", "t5_column"]
        eq(changed, want, "percentile %d against labels.csv, tau left out: %s" % (
            p, "only O1 (RECALL_FAILURE to LONG_TAIL, undetermined to training process)" if p >= 11 else "identical"))
    check(all("sweep_point" in r["confidence_reasons"] for r in srows if r["prompt_id"] == "biographies-00101"),
          "O1 carries sweep_point in all 16 sweep rows (Medium to 10, Low from 11)")

    print("4. labels_bio_tau.csv (Biographies' own tau, 10th percentile)")
    header, brows = read_csv(env.out / c2cli.BIO_NAME)
    eq(header, list(c2.LABEL_COLUMNS), "header is LABEL_COLUMNS (no percentile column)")
    eq(brows, expected_bio(), "every row at Biographies' tau")
    changed = {k: [c for c in diff_cols(head_by[k], b) if c != "tau"] for k, b in zip(KEYS, brows)}
    eq({k: v for k, v in changed.items() if v},
       {D2K: ["phase2_type", "t5_column"],
        D5K: ["confidence", "confidence_reasons", "coverage_downgrade", "f5_set", "intervention_point",
              "phase2_type", "t5_column"],
        O1K: ["confidence", "confidence_reasons", "coverage_downgrade", "f5_set", "intervention_point",
              "phase2_type", "t5_column"]},
       "against labels.csv, tau left out: only D2 (type), D5 and O1 (type, point, confidence)")
    eq(sorted({(b["model"], b["tau"]) for b in brows}), [(D, "3"), (O, "3"), (R, "2")],
       "tau column: OLMo 2 3, DCLM 3, RedPajama 2")
    b5 = brows[KEYS.index(D5K)]
    eq((b5["confidence"], b5["confidence_reasons"], b5["coverage_downgrade"]), ("Low", "absence_point; sweep_point", "2"),
       "D5 (point constant over the sweep, moves only at Biographies' tau): sweep_point fires there, from the "
       "headline sweep set plus its own point; DCLM's coverage_downgrade 2")

    print("5. t5.csv")
    header, trows = read_csv(env.out / c2cli.T5_NAME)
    eq(header, ["model", "regime", "t5_column", "slots"], "header")
    eq([(r["model"], r["regime"], r["t5_column"], r["slots"]) for r in trows],
       [(O, "S2", RF, "1"), (O, "S2", "no count row", "1"), (O, "S3", UNT, "1"), (O, "S1", "never wrong (no label)", "1"),
        (D, "S1", RF, "1"), (D, "S2", BH, "1"), (D, "S2", RF, "1"), (D, "single_mention", FA, "1"),
        (D, "S1", "never wrong (no label)", "1"),
        (R, "S1", UNT, "1"), (R, "S3", BL, "1")],
       "rows: MODELS order, regime, T5 order then `no count row`, then each model's never-wrong line")
    eq(sum(int(r["slots"]) for r in trows), 11, "total 11 = 9 labelled + 2 never wrong")
    eq(sum(int(r["slots"]) for r in trows if r["t5_column"] == "never wrong (no label)"), 2, "never wrong total 2")

    print("5b. f5_by_model.csv (F5's input, each model's points with their confidence)")
    header, frows = read_csv(env.out / c2cli.F5_MODEL_NAME)
    eq(header, c2cli.F5_MODEL_COLUMNS, "header: F5_MODEL_COLUMNS")
    eq([r["model"] for r in frows], [m for m in list(c2cli.MODELS) + ["all"] for _ in range(8)],
       "8 rows (6 points, the 2 open-band parts) per model in MODEL_INDEX order, then all")
    fmeta = json.loads((env.out / c2cli.META_NAME).read_text(encoding="utf-8"))
    eq([(r["point"], r["lower"], r["upper"], r["n"]) for r in frows if r["model"] == "all"],
       [(r["point"], str(r["lower"]), str(r["upper"]), str(r["n"])) for r in fmeta["f5"]["rows"]],
       "the all rows equal meta.json's F5 rows")
    eq(sum(int(r["n"]) for r in frows if r["model"] != "all" and r["point"] == cu.INFERENCE),
       int(next(r["n"] for r in frows if r["model"] == "all")), "the models' F5 slots add up to all's")
    tog = cu.TOGETHER
    check(all(int(r["lower"]) == int(r["high"]) + int(r["medium"]) + int(r["low"])
              for r in frows if r["high"] != "" and r["point"] != tog),
          "each point: the slots fixed there are its High + Medium + Low")
    eq({(r["model"], r["point"]): (r["high"], r["medium"], r["low"]) for r in frows if r["high"] == ""},
       {(m, b): ("", "", "") for m in list(c2cli.MODELS) + ["all"] for b in (c2.OPEN_BAND["S3"], c2.OPEN_BAND["S2"])},
       "the open band has no confidence")
    lab_conf = Counter((r["model"], r["confidence"]) for r in rows if r["confidence"])
    for m in c2cli.MODELS:
        got = {lvl: sum(int(r[lvl]) for r in frows if r["model"] == m and r["high"] != "" and r["point"] != tog)
               for lvl in ("high", "medium", "low")}
        eq(got, {"high": lab_conf[(m, "High")], "medium": lab_conf[(m, "Medium")], "low": lab_conf[(m, "Low")]},
           "%s: High, Medium, Low over its points equal labels.csv's" % m)

    print("6. meta.json")
    meta = json.loads((env.out / c2cli.META_NAME).read_text(encoding="utf-8"))
    check({"rule_version", "tau", "never_wrong", "f5", "sweep", "ordering_reversals", "sensitivity"} <= set(meta),
          "meta has rule_version, tau, never_wrong, f5, sweep, ordering_reversals, sensitivity")
    eq((meta["rule_version"], meta["confidence_rule"], meta["script_version"], meta["slots"], meta["labelled"]),
       (RV, c2.CONFIDENCE_RULE, c2cli.SCRIPT_VERSION, 11, 9), "rule, confidence rule, script version, 11 slots, 9 labelled")
    eq(meta["tau"], {"headline_percentile": 10,
                     "historical_events": {str(p): {i: float(t) for i, t in HE_TAU[p].items()} for p in SWEEP},
                     "biographies": {i: float(t) for i, t in BIO_TAU.items()}},
       "tau: the headline file at every percentile (decoy variant skipped), Biographies' at 10 (decoys skipped)")
    eq(meta["never_wrong"], {O: 1, D: 1, "olmo17-7b": 0, R: 0, "pythia-69b": 0}, "never_wrong per model")
    eq((meta["f5"]["n"], meta["f5"]["open"], meta["f5"]["open_groups"]), (7, 0, []),
       "f5 over 7 slots (S1 to S3, a count row, a wrong sample: D2 D4 D5 O1 O3 R1 R2), none open")
    eq([(r["point"], r["lower"], r["upper"], r["share_lower"], r["share_upper"], r["n"]) for r in meta["f5"]["rows"]],
       [(INF, 2, 2, "0.2857", "0.2857", 7), (TD, 1, 1, "0.1429", "0.1429", 7), (TP, 0, 0, "0.0000", "0.0000", 7),
        (cu.BOUNDARY, 0, 0, "0.0000", "0.0000", 7), (UND, 4, 4, "0.5714", "0.5714", 7),
        (cu.TOGETHER, 1, 1, "0.1429", "0.1429", 7),
        (c2.OPEN_BAND["S3"], 0, 0, "0.0000", "0.0000", 7), (c2.OPEN_BAND["S2"], 0, 0, "0.0000", "0.0000", 7)],
       "f5 rows: inference 2 (D2 R1), data 1 (D4), undetermined 4 (D5 O1 O3 R2), together 1, open bands 0")
    zero = {"t5": 0, "t5_all": 0, "f5": 0, "conf": 0}
    eq(meta["sweep"], {str(p): (zero if p <= 10 else {"t5": 1, "t5_all": 2, "f5": 1, "conf": 1}) for p in SWEEP},
       "sweep deltas: none to 10; from 11 O1's T5, F5 and confidence, plus O2's T5 in t5_all")
    pts_up, pts_down = list(range(11, 21)), list(range(5, 11))
    eq(meta["ordering_reversals"],
       {"F5 %s, lower" % cu.TOGETHER: [[O, D, pts_up, pts_down]], "F5 %s, upper" % cu.TOGETHER: [[O, D, pts_up, pts_down]]},
       "ordering reversals: only F5 together (OLMo 2 0 below DCLM 1/3 to 10, 1/2 above it from 11)")
    eq(meta["sensitivity"],
       {"t5": [[list(D2K), RF, LT], [list(D5K), RF, LT], [list(O1K), RF, LT]],
        "t5_all": [[list(D2K), RF, LT], [list(D5K), RF, LT], [list(O1K), RF, LT], [list(O2K), RF, LT]],
        "f5": [[list(D5K), UND, TP], [list(O1K), UND, TP]],
        "conf": [[list(D5K), "High", "Low"], [list(O1K), "Medium", "Low"]]},
       "sensitivity: T5 D2 D5 O1 (t5_all adds never-wrong O2), F5 D5 O1, confidence D5 O1")
    p = meta["provenance"]
    eq(p["walk"], {"targets": 1, "hits": 1, "phrases": 2}, "walk: 1 target (O3's c 1820), a hit on phrase 2")
    eq(p["verifier"], {"root": str(env.verifier.resolve()), "outputs": "outputs_filtered", "samples": 41,
                    "skipped_units": [], "claims_read": 41, "claims_kept": 40, "claims_dropped": 1},
       "verifier: 41 generations, D3's abstained sample dropped")
    eq(p["cache_path"], str(env.cache.resolve()), "cache path")
    eq({k: v["blob"] for k, v in p["inputs"].items()},
       {"claims": blob(env.claims), "counts": blob(env.counts), "tau": blob(env.tau), "Biographies tau": blob(env.bio_tau)},
       "input blobs (git hash-object of each file)")
    eq(p["decisions"], c2cli.DECISIONS, "decisions")

    print("7. summary.md")
    s = (env.out / c2cli.SUMMARY_NAME).read_text(encoding="utf-8")
    check(chr(0x2014) not in s and chr(0x2013) not in s, "no em or en dash")
    for h in ("## (a) T5: Phase 2 type by regime", "## (b) F5 range and open band", "## (c) Confidence (rule v3)",
              "## (d) The tau sweep (historical_events, 5% to 20%)", "## (e) Sensitivity: Biographies' own tau",
              "## (f) No LLM step for year slots", "## (g) Provenance"):
        check(h in s, "section %r" % h)
    heads = [s.index(h) for h in ("## (a)", "## (b)", "## (c)", "## (d)", "## (e)", "## (f)", "## (g)")]
    eq(heads, sorted(heads), "sections (a) to (g) in order")
    f2 = s.split("## (f2) Not labelled here")[1].split("## (g)")[0] if "## (f2) Not labelled here" in s else ""
    check(s.index("## (f)") < s.index("## (f2)") < s.index("## (g)") if f2 else False, "(f2) between (f) and (g)")
    check("item-level regimes only" in f2 and "historical_events enters only through tau" in f2,
          "(f2) the other seven tasks")
    lines = s.splitlines()

    def has(line, label):
        check(line in lines, label + ("" if line in lines else "  (missing line %r)" % line))

    check("(headline_406, 10th percentile): OLMo 2 2, DCLM 1, OLMo 1.7 1, RedPajama 2, Pythia 3." in s,
          "intro: the headline tau per model")
    check("Slots with a Wikidata year: 11 (S1 4, S2 4, S3 2, `single_mention` 1). Never-wrong S1 slots (no wrong sample: "
          "not a hallucination, no label): 2. Labelled slots: 9." in s,
          "intro: 11 slots by regime, 2 never wrong, 9 labelled")
    a = s.split("## (a)")[1].split("## (b)")[0]
    al = a.split("All models:")[1].split("OLMo 2 (`olmo2-7b`):")[0]
    eq([x for x in al.splitlines() if x.startswith("| ") and not x.startswith("| T5 column")],
       ["| %s | 0 | 1 | 0 | 0 | 1 |" % BH, "| %s | 0 | 0 | 1 | 0 | 1 |" % BL, "| %s | 0 | 0 | 0 | 1 | 1 |" % FA,
        "| %s | 1 | 2 | 0 | 0 | 3 |" % RF, "| %s | 1 | 0 | 1 | 0 | 2 |" % UNT, "| no count row | 0 | 1 | 0 | 0 | 1 |",
        "| all labelled | 2 | 4 | 2 | 1 | 9 |"], "(a) all models: T5 rows by regime, then no count row, then the total")
    ol = a.split("OLMo 2 (`olmo2-7b`):")[1].split("DCLM (`dclm-7b`):")[0]
    eq([x for x in ol.splitlines() if x.startswith("| ") and not x.startswith("| T5 column")],
       ["| %s | 0 | 1 | 0 | 0 | 1 |" % RF, "| %s | 0 | 0 | 1 | 0 | 1 |" % UNT, "| no count row | 0 | 1 | 0 | 0 | 1 |",
        "| all labelled | 0 | 2 | 1 | 0 | 3 |"], "(a) OLMo 2: O1, O3, O4 (O2 never wrong, not a T5 column)")
    for line in ("| OLMo 2 | 1 | 1 |", "| DCLM | 1 | 0 |", "| OLMo 1.7 | 0 | 0 |", "| RedPajama | 0 | 0 |",
                 "| Pythia | 0 | 0 |", "| all | 2 | 1 |"):
        has(line, "(a) own lines: %s" % line)
    has("- No count row (1: S2 1): a slot whose own or wrong year has no count row never reached the rules. Its own "
        "line, out of F5, not the residual bucket.",
        "(a) the 7(d) line")
    has("- Residual bucket (a slot with count rows that no rule typed): 0. Every slot with its count rows gets a type, a merged type or the "
        "untyped column, so the bucket is empty by construction for year slots.", "(a) the residual line: 0")
    check("Placement: Biographies' T5 columns and F5 stay in the body if the year judge's kappa is 0.667 or above, "
          "and go to the supplementary material unchanged below it." in s, "intro: the placement note")
    b2 = s.split("## (b2) F5 by model, with confidence")[1].split("## (c)")[0]
    eq([x for x in b2.splitlines() if x.startswith("| all |")],
       ["| all | at inference | 2 | 28.6% | 1 | 1 | 0 |", "| all | at training data | 1 | 14.3% | 0 | 1 | 0 |",
        "| all | undetermined | 4 | 57.1% | 2 | 2 | 0 |",
        "| all | training data and process together | 1 | 14.3% | 0 | 1 | 0 |"],
       "(b2) all models: the points with a slot, their share and confidence")
    b = s.split("## (b)")[1].split("## (b2)")[0]
    check("Over the 7 S1 to S3 slots" in b, "(b) over 7 slots")
    eq([x for x in b.splitlines() if x.startswith("| ") and not x.startswith("| Intervention")],
       ["| at inference | 28.6% | 2 |", "| at training data | 14.3% | 1 |", "| at training process | 0.0% | 0 |",
        "| training data and process together | 14.3% | 1 |", "| at the knowledge boundary | 0.0% | 0 |",
        "| undetermined | 57.1% | 4 |", "| %s | 0.0%% | 0 |" % c2.OPEN_BAND["S3"], "| %s | 0.0%% | 0 |" % c2.OPEN_BAND["S2"]],
       "(b) the F5 table and the open band")
    c_ = s.split("## (c)")[1].split("## (d)")[0]
    check(c2.CONFIDENCE_RULE in c_, "(c) the v3 rule text")
    has("Slots with a point: 7 (= F5's 7 less the 0 open). Blank: 2 (`single_mention`, the open band, no count row).",
        "(c) 7 with a point, 2 blank (D3, O4)")
    for line in ("| S1 | 1 | 1 | 0 | 0 |", "| S2 | 1 | 2 | 0 | 1 |", "| S3 | 1 | 1 | 0 | 0 |",
                 "| `single_mention` | 0 | 0 | 0 | 1 |", "| all | 3 | 4 | 0 | 2 |",
                 "| OLMo 2 | 1 | 1 | 0 |", "| DCLM | 2 | 1 | 0 |", "| OLMo 1.7 | 0 | 0 | 0 |", "| RedPajama | 0 | 2 | 0 |",
                 "| Pythia | 0 | 0 | 0 |",
                 "| absence_point | Low | 0 | 0 | 0 | 0 | 0 | 0 | n/a |", "| low_n | Low | 0 | 0 | 0 | 0 | 0 | 0 | n/a |",
                 "| sweep_point | Medium | 1 | 1 | 0 | 0 | 0 | 0 | 1 |",
                 "| s2_rho_near_half | Medium | 1 | 0 | 1 | 0 | 0 | 0 | 1 |",
                 "| s3_c_near_cut | Medium | 1 | 0 | 0 | 0 | 1 | 0 | 1 |",
                 "| s1_one_right | Medium | 0 | 0 | 0 | 0 | 0 | 0 | 0 |",
                 "| no_count_above_0 | Medium | 1 | 0 | 0 | 0 | 1 | 0 | 1 |"):
        has(line, "(c) %s" % line)
    has("- On S1, `no_count_above_0` (1 slots) means not phrase-checked, nothing found: the phrase check runs on S2 and "
        "S3 slots only (scope `s2s3`), so an S1 slot's `unknown` counts were never looked up.", "(c) the S1 no_count_above_0 line")
    has("- `fabrication_s1` (FABRICATION with S1, at inference and flagged): 0 labelled slots. The size of this cell "
        "measures how much the absence evidence misses. "
        "Typed FABRICATION S1 slots in this run: 1 (dclm-7b biographies-00201 birth), all never wrong, so none is "
        "labelled.", "(c) fabrication_s1 0: the only typed FABRICATION S1 slot (D1) is never wrong")
    has("- `s2_rho_near_half`: 1 slots (S2 slots typed BOTH_IN_DATA: 1). The slot that carried it under the earlier "
        "sample rule, redpajama-7b biographies-00665 death, is S1 here (a sample stating the right year is right "
        "moves it from S2 to S1).",
        "(c) s2_rho_near_half 1 (D4) and the 00665 slot named as S1")
    has("- Coverage downgrade, recorded and not applied (the two absence points are already Low; a gap in the spec): "
        "0 on 9 slots.", "(c) coverage downgrade 0 on all 9 at the headline")
    d_ = s.split("## (d)")[1].split("## (e)")[0]
    has("| Percentile | OLMo 2 | DCLM | OLMo 1.7 | RedPajama | Pythia | T5 column changes (of 9) | F5 point changes (of 7) "
        "| Confidence changes (of 9) | Type changes, all slots (of 11) |", "(d) header")
    for line in ("| 5% | 2 | 1 | 1 | 2 | 3 | 0 | 0 | 0 | 0 |", "| 10% (headline) | 2 | 1 | 1 | 2 | 3 | 0 | 0 | 0 | 0 |",
                 "| 11% | 4 | 1 | 1 | 2 | 3 | 1 | 1 | 1 | 2 |", "| 20% | 4 | 1 | 1 | 2 | 3 | 1 | 1 | 1 | 2 |"):
        has(line, "(d) %s" % line)
    check("Quantities: 21. With a reversal: 2." in d_, "(d) 21 quantities, 2 with a reversal")
    has("F5 shares whose order of models changes: F5 %s, lower, F5 %s, upper; every T5 column too." % (
        cu.TOGETHER, cu.TOGETHER), "(d) the reversal sentence as written (see the report on its wording)")
    up, down = ", ".join("%d%%" % x for x in pts_up), ", ".join("%d%%" % x for x in pts_down)
    for side in ("lower", "upper"):
        has("| F5 %s, %s | OLMo 2, DCLM | %s | %s |" % (cu.TOGETHER, side, up, down), "(d) reversal row, %s" % side)
    e = s.split("## (e)")[1].split("## (f)")[0]
    check("10th percentile): OLMo 2 3, DCLM 3, OLMo 1.7 1, RedPajama 2, Pythia 3. Headline: OLMo 2 2, DCLM 1, "
          "OLMo 1.7 1, RedPajama 2, Pythia 3." in e, "(e) both taus")
    for line in ("- T5 column changes: 3 of 9 labelled slots (4 of all 11 slots, counting the never-wrong slots by "
                 "their pre-label type).",
                 "- F5 point changes: 2 of 7 F5 slots.", "- Confidence changes: 2 of 9 labelled slots.",
                 "| dclm-7b biographies-00202 birth | S1 | no | %s | %s | %s | %s |" % (RF, LT, INF, INF),
                 "| dclm-7b biographies-00205 birth | S2 | no | %s | %s | %s | %s |" % (RF, LT, UND, TP),
                 "| olmo2-7b biographies-00101 birth | S2 | no | %s | %s | %s | %s |" % (RF, LT, UND, TP),
                 "| olmo2-7b biographies-00102 birth | S1 | yes | %s (pre-label) | %s (pre-label) | none (never wrong) "
                 "| none (never wrong) |" % (RF, LT)):
        has(line, "(e) %s" % line)
    eq(sum(1 for x in e.splitlines() if x.startswith("| ") and "biographies-" in x), 4, "(e) table: 4 slots")
    g = s.split("## (g)")[1]
    has("- phrase check: cache `%s` (read-only), 1 targets, 1 with a hit, 2 phrases walked, 0 requests."
        % env.cache.resolve(), "(g) the phrase check line")
    has("- verifier output copy: `%s` (`outputs_filtered`), 41 samples, skipped units: none; claims kept 40 of 41 (1 dropped)."
        % env.verifier.resolve(), "(g) the verifier output line")
    check("- git head: `" in g and "script version 1, rule `%s`." % RV in g, "(g) git head, script version, rule")

    print("8. rerun, --force, determinism")
    code, out, err = run(env.args())
    eq(code, 1, "a second run without --force: exit 1")
    check("already has" in err and "--force" in err and "Traceback" not in err, "the refusal names --force")
    eq(snapshot(env.out), first, "outputs untouched")
    out2 = env.d / "out2"
    code, out, err = run(env.args(out=out2))
    eq(code, 0, "a run into another --out: exit 0")
    eq({n: (out2 / n).read_bytes() for n in (c2cli.LABELS_NAME, c2cli.SWEEP_NAME, c2cli.BIO_NAME, c2cli.T5_NAME)},
       {n: first[n] for n in (c2cli.LABELS_NAME, c2cli.SWEEP_NAME, c2cli.BIO_NAME, c2cli.T5_NAME)},
       "the four CSVs are byte-identical across runs")
    one = env.d / "out3"
    one.mkdir()
    (one / c2cli.LABELS_NAME).write_bytes(b"planted\n")
    code, out, err = run(env.args(out=one))
    eq((code, listing(one), (one / c2cli.LABELS_NAME).read_bytes()), (1, [c2cli.LABELS_NAME], b"planted\n"),
       "one planted output, no --force: exit 1, nothing written, the file untouched")
    check("already has labels.csv; pass --force to overwrite" in err, "the refusal names the file")
    code, out, err = run(env.args("--force", out=one))
    eq((code, listing(one), (one / c2cli.LABELS_NAME).read_bytes()), (0, sorted(c2cli.OUTPUTS), first[c2cli.LABELS_NAME]),
       "--force overwrites it and writes the other five")

    print("9. refusals: exit 1, nothing written, existing outputs untouched (each with --force)")

    def refused(argv, needle, label, fresh_out=True):
        code, out, err = run(argv + ["--force"])
        eq(code, 1, "%s: exit 1" % label)
        check(needle in err and "Traceback" not in err, "%s: stderr says %r (%r)" % (label, needle, err.strip()[:300]))
        eq(snapshot(env.out), first, "%s: existing outputs byte-identical, no temporary file" % label)
        if fresh_out:
            nowhere = env.d / "never_out"
            argv2 = list(argv)
            argv2[argv2.index("--out") + 1] = str(nowhere)
            code2, _, _ = run(argv2)
            eq((code2, nowhere.exists()), (1, False), "%s: with a fresh --out, nothing is created" % label)

    nocache = env.d / "nocache" / "phrase_counts.sqlite"
    refused(env.args(cache=nocache), "no phrase cache at", "no cache file")
    eq((nocache.exists(), nocache.parent.exists()), (False, False), "no cache file: nothing created")
    named = env.d / "wc1" / "infinigram.sqlite"
    refused(env.args(cache=named), "infinigram.sqlite is the window cache", "a cache named infinigram.sqlite")
    eq(named.exists(), False, "infinigram.sqlite: not created")
    wc = env.d / "wc2" / "shard.sqlite"
    c = infinigram.Cache(str(wc), lock=False)
    try:
        c.conn.execute("INSERT INTO windows (index_name, query_string, form, s, rank, spans_json, fetched_at) "
                       "VALUES ('i', 'q', 'plain', 0, 0, '[]', 'now')")
        c.conn.commit()
    finally:
        c.close()
    wb, wl = wc.read_bytes(), listing(wc.parent)
    refused(env.args(cache=wc), "so it is a window cache", "a cache whose windows table holds a row")
    eq((wc.read_bytes() == wb, listing(wc.parent)), (True, wl), "window cache: byte-identical, nothing beside it")

    def tau_file(name, columns, rows):
        path = env.d / "taus" / name
        write_csv(path, columns, rows)
        return path

    refused(env.args(tau=tau_file("no20.csv", HE_COLUMNS, he_rows(lambda p, i: p == 20))),
            "headline_406 percentiles [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], expected 5 to 20",
            "he tau without percentile 20")
    refused(env.args(tau=tau_file("no20pile.csv", HE_COLUMNS, he_rows(lambda p, i: p == 20 and i == PILE))),
            "tau: missing tau row at percentile 20 (variant 'headline_406') for ['%s']" % PILE,
            "he tau missing one index at percentile 20")
    refused(env.args(expected_tau=EXPECTED_TAU.replace("%s=2" % OLMO, "%s=1" % OLMO)),
            "tau: unexpected value of tau at percentile 10", "he tau headline differs from --expected-tau")
    refused(env.args(bio_tau=tau_file("bio_variant.csv", ["variant"] + BIO_COLUMNS,
                                      [dict(r, variant="headline_406") for r in bio_rows()])),
            "variant None but a tau row has a variant column", "bio tau with a variant column")
    refused(env.args(expected_bio_tau=EXPECTED_BIO_TAU.replace("%s=3" % DCLM, "%s=2" % DCLM)),
            "tau: unexpected value of tau at percentile 10", "bio tau differs from --expected-bio-tau")
    partial = env.d / "pc2" / "phrase_counts.sqlite"
    fill_cache(partial, {(OLMO, TARGET_PHRASES[0]): 0})
    pb = partial.read_bytes()
    refused(env.args(cache=partial), "1 target(s) reach a phrase not in the cache (a request would be needed):\n  "
            "%s\t%s" % (OLMO, TARGET_PHRASES[1]), "a phrase the walk needs is not in the cache")
    eq(partial.read_bytes(), pb, "the partial cache is byte-identical")
    refused(env.args(verifier_root=env.d / "nowhere"), "no verifier output copy at %s" % (env.d / "nowhere"), "a missing --verifier-root")
    refused(env.args(verifier_root=None), "no verifier output copy at %s" % cli.DEFAULT_VERIFIER_ROOT,
            "no --verifier-root (the default, pointed at a temp folder that does not exist)")
    short = env.d / "short_verifier"
    gone = "%s|biographies-00302|1" % R
    write_verifier(short, [x for x in CLAIMS if x["generation_id"] != gone], outcomes={D3_ABSTAINED: "abstention"})
    refused(env.args(verifier_root=short), "%s: no sidecar row for this claim's generation" % gone,
            "a claim with no sidecar row")
    code, out, err = run(env.args(cache=None))
    eq((code, "--cache" in err), (2, True), "no --cache: exit 2 (argparse)")
    eq(snapshot(env.out), first, "outputs untouched after all refusals")
    eq((env.cache.read_bytes() == cache_before, listing(env.cache.parent)), (True, cache_listing),
       "the phrase cache untouched after every run")

    print("10. all or nothing: a write failure on the 3rd file")
    real = cli._write_file
    calls = []

    def failing(path, payload):
        calls.append(path)
        if len(calls) == 3:
            Path(path).write_bytes(b"partial")
            raise OSError("disk full (test)")
        real(path, payload)

    cli._write_file = failing
    try:
        code, out, err = run(env.args("--force"))
    finally:
        cli._write_file = real
    eq(code, 1, "exit 1")
    check("disk full (test)" in err and "Traceback" not in err, "one error line, no traceback")
    eq(snapshot(env.out), first, "prior outputs byte-identical, the temporary files removed")
finally:
    cli.HttpTransport, infinigram.HttpTransport = old["http"]
    cli.DEFAULT_VERIFIER_ROOT = old["verifier"]
    c2cli.DEFAULT_OUT = old["out"]
    if old["lock"] is None:
        os.environ.pop(infinigram.LOCK_DIR_ENV, None)
    else:
        os.environ[infinigram.LOCK_DIR_ENV] = old["lock"]
    tmp.cleanup()
eq((cli.HttpTransport, infinigram.HttpTransport, cli.DEFAULT_VERIFIER_ROOT, c2cli.DEFAULT_OUT),
   (old["http"][0], old["http"][1], old["verifier"], old["out"]), "module patches restored")
eq(os.environ.get(infinigram.LOCK_DIR_ENV), old["lock"], "LOCK_DIR_ENV restored")

# ---------------------------------------------------------------------------
print()
if failures:
    print("%d FAILED:" % len(failures))
    for f in failures:
        print("  - " + f)
    sys.exit(1)
print("all passed")
