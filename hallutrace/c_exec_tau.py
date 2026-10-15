"""hallutrace/c_exec_tau.py  |  historical_events tau with the counting rule (pure, no I/O)

The CLI (scripts/c_exec_tau.py) reads the window cache read-only, counts every
historical_events name once with `pilot_counts.count_grid`, keeps the rule's cell
(`bio_counts.select_cell(rows, bio_counts.RULE)`: near_or_cue / full / no_stripped) and
hands the cell rows and the cache's entity rows to this module.

Name sets, from ACTIVE entity rows only (never a hardcoded list):

  all        every prompt name with an active query string
  short      names with an active query string whose source is `stripped` (a Wikipedia
             disambiguator, "George Ashby (MP)"): the rule counts them without that short form
  ambiguous  names whose active rows have `ambiguous_name` truthy (the alias table
             gives the name two or more QIDs)

Variants, in this order; the numbers are the expected
sizes, membership always comes from the sets:

  headline_406      all minus short (the headline set)
  with_short_444    all (sensitivity row)
  no_ambiguous_265  all minus short minus ambiguous (sensitivity row)

Tau per variant is `bio_counts.tau_rows` over that variant's rows of the cell (nearest
rank over n_c > 0, every cutoff 5% to 20%, over the correct facts). Both
definitions are kept side by side and neither is the headline yet: `tau` (docs units
only) and `tau_with_est` (also the sampled pools' est_mentions). An `unknown` fact (a
sampled pool with no match) enters neither; a `zero` fact never enters `tau`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from . import bio_counts as bc
from . import pilot_counts as pc

VARIANTS = ("headline_406", "with_short_444", "no_ambiguous_265")
HEADLINE = "headline_406"
TAU_P = bc.TAU_P  # 10
# Expected set sizes on the 1 Oct cache; the CLI refuses to run when they differ.
# `overlap` is |short & ambiguous| and must be 0.
EXPECTED_SIZES = {"all": 444, "short": 38, "ambiguous": 141, "overlap": 0}
EXTRA_COLUMNS = ["short_form", "in_headline"]
COUNTS_COLUMNS = pc.CSV_COLUMNS + EXTRA_COLUMNS
TAU_COLUMNS = ["variant", "index", "model", "percentile", "tau", "tau_with_est", "facts",
               "counted", "zero", "unknown", "docs_above_0", "est_mentions", "year_conflict"]


def name_sets(entities: Iterable[dict]) -> Dict[str, List[str]]:
    """{"all": [...], "short": [...], "ambiguous": [...]}, each a sorted list of prompt
    names, from ACTIVE entity rows only (`pilot_counts._active`). `short`: names with an
    active row whose `source == "stripped"`; an inactive stripped row does not count.
    `ambiguous`: names with an active row whose `ambiguous_name` is truthy
    (`pilot_counts._truthy`, so 1, "1", True count and 0, "0", "", None do not)."""
    every, short, amb = set(), set(), set()
    for e in entities:
        if not pc._active(e):
            continue
        name = e["prompt_name"]
        every.add(name)
        if e.get("source") == "stripped":
            short.add(name)
        if pc._truthy(e.get("ambiguous_name")):
            amb.add(name)
    return {"all": sorted(every), "short": sorted(short), "ambiguous": sorted(amb)}


def set_sizes(sets: Dict[str, List[str]]) -> Dict[str, int]:
    """{"all", "short", "ambiguous", "overlap"} sizes (overlap = names in both short and
    ambiguous), plus the size of each variant under its VARIANTS name."""
    out = {"all": len(sets["all"]), "short": len(sets["short"]),
           "ambiguous": len(sets["ambiguous"]),
           "overlap": len(set(sets["short"]) & set(sets["ambiguous"]))}
    for variant, names in variant_names(sets).items():
        out[variant] = len(names)
    return out


def check_sizes(sets: Dict[str, List[str]], expected: Dict[str, int] = EXPECTED_SIZES) -> List[str]:
    """Problems (empty list = sizes as expected): one line per key of `expected` whose
    size differs, e.g. "short: 37 names, expected 38"."""
    got = set_sizes(sets)
    return ["%s: %d names, expected %d" % (k, got[k], want)
            for k, want in expected.items() if got[k] != want]


def variant_names(sets: Dict[str, List[str]]) -> Dict[str, set]:
    """{variant: set of prompt names} for every variant in VARIANTS order."""
    every, short, amb = set(sets["all"]), set(sets["short"]), set(sets["ambiguous"])
    by_name = {"headline_406": every - short, "with_short_444": set(every),
               "no_ambiguous_265": every - short - amb}
    return {v: by_name[v] for v in VARIANTS}


def as_csv_row(row: dict) -> dict:
    """The row as a CSV round trip would give it: None -> "", every other value str(v).
    `bio_counts.tau_rows` counts `year_conflict == "1"`, so in-memory count_grid rows
    (ints) must go through this before tau_rows. Idempotent on rows already all str."""
    return {k: "" if v is None else str(v) for k, v in row.items()}


def tau_table(cell_rows: List[dict], sets: Dict[str, List[str]],
              percents: Iterable[int] = bc.PERCENTS) -> List[dict]:
    """For each variant in VARIANTS order: `bio_counts.tau_rows` over that variant's
    rows (prompt_name in the variant) of the rule's cell, each output row prefixed with
    {"variant": name} (key order TAU_COLUMNS). Rows go through as_csv_row first, so
    count_grid rows and CSV rows give the same table. Both `tau` and `tau_with_est` are
    kept as tau_rows returns them (a number, or None when the population is empty)."""
    percents = list(percents)
    rows = [as_csv_row(r) for r in cell_rows]
    out = []
    for variant, names in variant_names(sets).items():
        part = [r for r in rows if r["prompt_name"] in names]
        for t in bc.tau_rows(part, percents):
            out.append(dict({"variant": variant}, **t))
    return out


def annotate(cell_rows: List[dict], sets: Dict[str, List[str]]) -> List[dict]:
    """Copies of the cell rows (order kept) with `short_form` (1 if the name is in
    short, else 0) and `in_headline` (1 if the name is in the headline variant, else 0)
    added; columns COUNTS_COLUMNS. `ambiguous_name` is already a count_grid column."""
    short = set(sets["short"])
    headline = variant_names(sets)[HEADLINE]
    return [dict(r, short_form=int(r["prompt_name"] in short),
                 in_headline=int(r["prompt_name"] in headline)) for r in cell_rows]


def _g(v) -> str:
    return "n/a" if v is None else "%g" % v


def _rows_of(tau: List[dict], variant: str, percentile=None) -> List[dict]:
    return [t for t in tau if t["variant"] == variant
            and (percentile is None or t["percentile"] == percentile)]


def summary_markdown(tau: List[dict], sets: Dict[str, List[str]], provenance: dict) -> str:
    """he_tau_summary.md. Headed "# historical_events tau". Sections:

    (a) both taus at TAU_P per model for the headline set, side by side, with a line
        stating plainly that neither `tau` nor `tau_with_est` is chosen as the headline
        definition yet;
    (b) the 5% to 20% sweep for the headline set, both taus (one row per percentile,
        a tau and a tau_with_est column per model);
    (c) the three variants at TAU_P, both taus, with the populations per index
        (`docs_above_0`, `est_mentions`, `unknown`, `zero`);
    (d) the name sets: sizes (all, short, ambiguous, overlap, each variant) and the
        short-form names listed;
    (e) provenance from `provenance` (keys cache_path, cache_size_bytes, cache_sha256,
        git_head, script_version, written_at, run_seconds, cell_rows), the rule
        (bio_counts.RULE), and the decisions behind it (headline without the 38 short-form
        names; the 141 ambiguous names kept).

    Numbers, names and years only; no corpus text. No em dashes. tau None shows "n/a";
    numbers are formatted "%g"."""
    sizes = set_sizes(sets)
    labels = list(pc.INDEX_LABELS.values())
    pv = provenance
    out = ["# historical_events tau", ""]
    out += ["Nearest-rank percentile over the correct facts with n_c above 0, "
            "one value per model. `tau` counts docs units only; `tau_with_est` also counts the "
            "sampled pools' est_mentions. `unknown` facts enter neither.", ""]

    # (a) both taus at TAU_P, headline set
    out += ["## (a) Both taus at %d%%, headline set (%s, %d names)"
            % (TAU_P, HEADLINE, sizes[HEADLINE]), ""]
    out += ["Neither `tau` nor `tau_with_est` is chosen as the headline definition yet; both "
            "are shown side by side.", ""]
    out += pc._table(["Model", "tau", "tau_with_est"],
                     [[t["model"], _g(t["tau"]), _g(t["tau_with_est"])]
                      for t in _rows_of(tau, HEADLINE, TAU_P)])

    # (b) the sweep, headline set
    out += ["## (b) Sweep over the cutoff, headline set (%s)" % HEADLINE, ""]
    head = ["Percentile"] + [m + " " + k for m in labels for k in ("tau", "tau_with_est")]
    body = []
    for p in sorted({t["percentile"] for t in _rows_of(tau, HEADLINE)}):
        by_model = {t["model"]: t for t in _rows_of(tau, HEADLINE, p)}
        row = ["%d%%" % p]
        for m in labels:
            t = by_model.get(m)
            row += [_g(t["tau"]) if t else "n/a", _g(t["tau_with_est"]) if t else "n/a"]
        body.append(row)
    out += pc._table(head, body)

    # (c) the variants at TAU_P with their populations
    out += ["## (c) Variants at %d%%, with the populations" % TAU_P, ""]
    out += ["`docs_above_0`: facts with n_c above 0 in docs units (the tau population). "
            "`est_mentions`: counted facts from sampled pools. `unknown`: no match from a "
            "sampled pool. `zero`: fully fetched, nothing matched.", ""]
    for variant in VARIANTS:
        out += ["### %s (%d names)" % (variant, sizes[variant]), ""]
        out += pc._table(["Model", "tau", "tau_with_est", "facts", "docs_above_0",
                          "est_mentions", "unknown", "zero"],
                         [[t["model"], _g(t["tau"]), _g(t["tau_with_est"]), t["facts"],
                           t["docs_above_0"], t["est_mentions"], t["unknown"], t["zero"]]
                          for t in _rows_of(tau, variant, TAU_P)])

    # (d) the name sets
    out += ["## (d) Name sets", ""]
    out += pc._table(["Set", "Names"],
                     [["all", sizes["all"]], ["short", sizes["short"]],
                      ["ambiguous", sizes["ambiguous"]],
                      ["overlap (short and ambiguous)", sizes["overlap"]]]
                     + [[v, sizes[v]] for v in VARIANTS])
    out += ["Short-form names (%d), counted without the disambiguator-stripped string:"
            % sizes["short"], ""]
    out += ["- " + n for n in sets["short"]] + [""]

    # (e) provenance
    out += ["## (e) Provenance", ""]
    out += ["- cache: %s" % pv.get("cache_path"),
            "- cache size (bytes): %s" % pv.get("cache_size_bytes"),
            "- cache SHA-256: %s" % pv.get("cache_sha256"),
            "- git head: %s" % pv.get("git_head"),
            "- script version: %s" % pv.get("script_version"),
            "- written at: %s" % pv.get("written_at"),
            "- run seconds: %s" % pv.get("run_seconds"),
            "- cell rows: %s" % pv.get("cell_rows"),
            "- rule (counting cell): matcher %s, half_width %s, strings %s"
            % (bc.RULE["matcher"], bc.RULE["half_width"], bc.RULE["strings"]),
            "- near: %d characters; cap_full: %d windows" % (pc.NEAR, pc.CAP_FULL), ""]
    out += ["Decisions:", "",
            "- tau is the nearest-rank percentile of n_c over facts with n_c above 0; "
            "cutoffs 5%% to 20%%, %d%% shown first." % TAU_P,
            "- the facts are the correct facts (birth and death years).",
            "- count with the near_or_cue matcher, the full window and no stripped "
            "short forms.",
            "- the headline set leaves out the short-form names "
            "(%d names)." % sizes["short"],
            "- the ambiguous names are kept (%d names); %s shows the "
            "set without them." % (sizes["ambiguous"], VARIANTS[2]),
            "- an unknown fact (a sampled pool with no match) enters neither tau.",
            ""]
    return "\n".join(out)
