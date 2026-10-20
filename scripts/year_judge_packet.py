r"""The gold check of the Biographies year judge (Wikidata), one annotation packet.

T5 and F5 judge a stated birth or death year right or wrong by an exact match against the
person's Wikidata year (`hallutrace/bio_years.py`), not by OpenFActScore, whose quarantine
(kappa 0.15) measures something else. This draws the year claims that judge is gated on
and writes them as an annotation packet, so `scripts/agreement.py` scores it unchanged.

    modal volume get hallutrace-data verifier/generations/inputs/<model>/<unit>.csv <root>\inputs\<model>\<unit>.csv
    .\.venv\Scripts\python.exe scripts\year_judge_packet.py <root>

<root>\inputs\<model>\<unit_id>.csv are local copies of the Biographies verifier input files, for
the response text; `--list-units` prints the ones the draw needs, before any download.

Draw (seeded, `random.Random(SEED)`): from `results/bio_year_claims.csv`, the claims of
sampled generations (sample_index 1 to 5, as the slot regimes use) whose person has a Wikidata
year, one per (generation_id, kind, stated_years) since repeated atoms are one decision of the
judge; N_EACH the judge calls right and N_EACH it calls wrong, in file order, then shuffled
together so the sheet does not group them.

Writes annotation/verifier_check/biographies_years/: annotator.csv (blank sheet, no verdict and no
Wikidata year), key.csv (the judge's verdict as `verifier_label`: supported when the year
matches Wikidata, hallucinated otherwise; `flagged` 1 for hallucinated), manifest.json and
README.md. Refuses to overwrite without --force. Read-only otherwise, no network.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLAIMS = ROOT / "results" / "bio_year_claims.csv"
OUT = ROOT / "annotation" / "verifier_check" / "biographies_years"
SEED = 20261009
N_EACH = 25
TASK = "biographies_years"
QUESTION = ("Does the response state this year as the person's {kind} year, and is it the "
            "right year?")
WIKI = "https://www.wikidata.org/wiki/Special:GoToLinkedPage/enwiki/%s"
FILES = ("annotator.csv", "key.csv", "manifest.json", "README.md")
SHEET_COLUMNS = ["item_id", "name", "kind", "wikipedia", "context_shown", "unit_to_judge",
                 "question", "annotator_label", "tag", "notes"]
KEY_COLUMNS = ["item_id", "generation_id", "model", "sample_index", "unit_id", "task",
               "verifier_label", "flagged", "stated_years", "wikidata_years", "own_qid",
               "openfactscore_label", "source_file"]


def population(claims):
    """The judge's decisions the gate covers: sampled generations, person with a Wikidata year,
    one per (generation_id, kind, stated_years). The decomposer can repeat a year atom within
    one response (up to 8 copies); the judge decides once, so the first row in file order
    stands for its copies, and a repeated atom is not drawn more often."""
    seen, out = set(), []
    for c in claims:
        if c["sample_index"] == "0" or not c["wikidata_years"].strip():
            continue
        k = (c["generation_id"], c["kind"], c["stated_years"])
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def draw(claims, seed=SEED, n_each=N_EACH):
    """n_each judged right and n_each judged wrong, then shuffled together."""
    pop = population(claims)
    right = [c for c in pop if c["wikidata_match"] == "True"]
    wrong = [c for c in pop if c["wikidata_match"] == "False"]
    if len(right) < n_each or len(wrong) < n_each:
        raise ValueError("population too small: %d right, %d wrong" % (len(right), len(wrong)))
    rng = random.Random(seed)
    picked = rng.sample(right, n_each) + rng.sample(wrong, n_each)
    rng.shuffle(picked)
    return picked, {"right": len(right), "wrong": len(wrong)}


def units_needed(picked):
    return sorted({"%s/%s.csv" % (c["model"], c["unit_id"]) for c in picked})


def responses(root, picked):
    """{generation_id: response text} from <root>/inputs/<model>/<unit_id>.csv."""
    out = {}
    for u in units_needed(picked):
        path = os.path.join(root, "inputs", *u.split("/"))
        if not os.path.exists(path):
            raise FileNotFoundError("missing input file %s (see --list-units)" % path)
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                out[r["generation_id"]] = r["response"]
    return out


def rows(picked, texts):
    sheet, key = [], []
    for i, c in enumerate(picked, 1):
        item = "%s-%03d" % (TASK, i)
        if c["generation_id"] not in texts:
            raise KeyError("no response for %s" % c["generation_id"])
        sheet.append({"item_id": item, "name": c["name"], "kind": c["kind"],
                      "wikipedia": WIKI % c["own_qid"], "context_shown": texts[c["generation_id"]],
                      "unit_to_judge": c["atom"], "question": QUESTION.format(kind=c["kind"]),
                      "annotator_label": "", "tag": "", "notes": ""})
        right = c["wikidata_match"] == "True"
        key.append({"item_id": item, "generation_id": c["generation_id"], "model": c["model"],
                    "sample_index": c["sample_index"], "unit_id": c["unit_id"], "task": TASK,
                    "verifier_label": "supported" if right else "hallucinated",
                    "flagged": "0" if right else "1", "stated_years": c["stated_years"],
                    "wikidata_years": c["wikidata_years"], "own_qid": c["own_qid"],
                    "openfactscore_label": c["label"],
                    "source_file": "%s/%s.csv" % (c["model"], c["unit_id"])})
    return sheet, key


def manifest(picked, pop):
    total = pop["right"] + pop["wrong"]
    return {
        "task": TASK, "seed": str(SEED),
        "labels": ["hallucinated", "supported", "unverifiable"],
        "question": QUESTION.format(kind="birth or death"),
        "measure": "kappa",
        "decision": "the year judge is gated on its own gold check",
        "judge": "exact match of the stated year against the person's Wikidata year "
                 "(hallutrace/bio_years.py), as T5 and F5 use it",
        "stats": {"pool": total, "pool_flagged": pop["wrong"],
                  "base_rate": round(pop["wrong"] / total, 4),
                  "drawn": len(picked), "drawn_flagged": N_EACH,
                  "sampling_ratio": round(N_EACH / len(picked), 4),
                  "drawn_from": "results/bio_year_claims.csv, sample_index 1 to 5, "
                                "Wikidata year present, one per (generation_id, kind, "
                                "stated_years)"},
        "sources": units_needed(picked),
    }


README = """# Biographies year judge: gold check

One annotator. The question for every row: **does the response state this year as
the person's birth (or death) year, and is it the right year?** Read `context_shown` (the
model's response) and `unit_to_judge` (the year claim extracted from it), and look the person
up through the `wikipedia` link (it opens the English Wikipedia article of the Wikidata item
the pipeline resolved the name to). Do not open `key.csv` until every row is labelled.

`annotator_label`:
- `supported`: the response states this year as this person's birth (death) year, and it is
  the right year.
- `hallucinated`: the response states this year as this person's birth (death) year, and it is
  wrong.
- `unverifiable`, with `tag` = `not a claim`: the response does not state it as this person's
  birth (death) year: the year belongs to another person (a parent, another prompt's name), to
  no person, or to something else (a publication, an event). A same-named different person, or
  a relative confused with the subject but named as the subject, counts as the subject, as in
  the other-person check (`results/bio_other_person_sample.csv`).

A year the response copies from the prompt (for example a name such as "Felipe (footballer,
born 1977)") still counts as stated by the response.

Copy `annotator.csv` to `annotator_a.csv`, label it, commit it. Score, with your file as
both A and B (one annotator, so gold is your label and the inter-annotator line means nothing):

    .\\.venv\\Scripts\\python.exe scripts\\agreement.py compute annotation\\verifier_check\\biographies_years --a annotation\\verifier_check\\biographies_years\\annotator_a.csv --b annotation\\verifier_check\\biographies_years\\annotator_a.csv

The gate is the verifier kappa on all rows: accept at 0.800, tentative from 0.667,
quarantine below. At 0.667 or above T5's Biographies columns and F5 stay in the body; below,
they go to the supplementary material.
"""


def write(out_dir, sheet, key, man, force=False):
    os.makedirs(out_dir, exist_ok=True)
    existing = [f for f in FILES if os.path.exists(os.path.join(out_dir, f))]
    if existing and not force:
        raise FileExistsError("%s already has %s; pass --force" % (out_dir, ", ".join(existing)))
    for name, cols, data in (("annotator.csv", SHEET_COLUMNS, sheet), ("key.csv", KEY_COLUMNS, key)):
        with open(os.path.join(out_dir, name), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, indent=2)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(README)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", nargs="?", help="folder holding inputs/<model>/<unit_id>.csv")
    p.add_argument("--claims", default=str(CLAIMS))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--list-units", action="store_true", help="print the input files the draw needs")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    with open(args.claims, encoding="utf-8", newline="") as f:
        claims = list(csv.DictReader(f))
    picked, pop = draw(claims)
    if args.list_units:
        print("\n".join(units_needed(picked)))
        return 0
    if not args.root:
        p.error("root is required unless --list-units")
    try:
        sheet, key = rows(picked, responses(args.root, picked))
        write(args.out, sheet, key, manifest(picked, pop), args.force)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    print("%d items (%d judged right, %d wrong) from %d right / %d wrong -> %s"
          % (len(sheet), N_EACH, N_EACH, pop["right"], pop["wrong"], args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
