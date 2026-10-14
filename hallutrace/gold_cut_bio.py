"""The gold cut for Biographies, on top of gold_cut.py.

One unit is one atomic fact, judged against the person's Wikipedia article, "Is this
fact about the person supported by the reference source?". The annotator sees the atom
and the article, never the model's response.

Where the pieces come from (all under the verification output root on hallutrace-data):

    outputs_filtered/{model}/{unit}.csv   HALoGEN's Biographies output, with the atoms
                                          Gemma's generator invented removed
                                          (hallutrace/bio_filter.py). The units are drawn
                                          from here, so annotators never see those atoms.
    inputs/{model}/{unit}.csv             the verifier input, which carries
                                          generation_id; the output does not, so rows
                                          are joined back by name (unique per unit)
    enwiki-20230401.db                    the article, from the same index OpenFActScore
                                          retrieved from (`documents(title, text)`,
                                          paragraphs joined by SPECIAL_SEPARATOR)

The verifier's label is OpenFActScore's verdict: True is supported, False hallucinated.
`would_drop_23` is always 0 here: the units come from the filtered output.
"""
from __future__ import annotations

import csv
import glob
import os
import sqlite3
from typing import Dict, Iterable, List, Tuple

from hallutrace import bio_filter as bfl
from hallutrace import gold_cut as gc

TASK = "biographies"
QUESTION = "Is this fact about the person supported by the reference source?"
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"

gc.QUESTIONS.setdefault(TASK, QUESTION)


def _names_to_ids(input_csv: str) -> Dict[str, str]:
    with open(input_csv, encoding="utf-8", newline="") as f:
        return {r["name"]: r["generation_id"] for r in csv.DictReader(f)}


def build_units(filtered_dir: str, inputs_dir: str) -> List[gc.Unit]:
    """Every kept atom of every Biographies unit output, as a gold_cut Unit.

    `context` holds the person's name for now; `attach_articles` replaces it with the
    article once the draw has picked the units, so only those articles are read.
    """
    csv.field_size_limit(10 ** 9)
    units: List[gc.Unit] = []
    for path in sorted(glob.glob(os.path.join(filtered_dir, "*", "*.csv"))):
        if not os.path.getsize(path):
            continue
        model = os.path.basename(os.path.dirname(path))
        unit_file = os.path.basename(path)
        ids = _names_to_ids(os.path.join(inputs_dir, model, unit_file))
        source = os.path.join(model, unit_file)
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                gid = ids[row["name"]]
                for i, (atom, supported) in enumerate(bfl.parse_atoms(row["atoms"])):
                    units.append(gc.Unit("%s:%d" % (gid, i), gid, TASK, row["name"],
                                         atom.strip(), gc.SUPPORTED if supported else gc.HALLUCINATED,
                                         False, source))
    return units


def article(conn: sqlite3.Connection, title: str) -> str:
    """The article as plain paragraphs, the way OpenFActScore's DocDB splits it."""
    got = conn.execute("SELECT text FROM documents WHERE title = ?", (title,)).fetchall()
    if len(got) != 1:
        return ""
    paras = [p.replace("<s>", "").replace("</s>", "").strip() for p in got[0][0].split(SPECIAL_SEPARATOR)]
    return "\n\n".join(p for p in paras if p)


def attach_articles(units: Iterable[gc.Unit], db_path: str) -> Tuple[int, List[str]]:
    """Replace each unit's context (the name) with "Name\\n\\narticle". Read-only.

    Returns (articles found, names with no article). A unit with no article keeps a
    note saying so, so the annotator marks it unverifiable rather than guessing.
    """
    conn = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    cache: Dict[str, str] = {}
    missing = []
    try:
        for u in units:
            name = u.context
            if name not in cache:
                cache[name] = article(conn, name)
                if not cache[name]:
                    missing.append(name)
            text = cache[name] or "(No Wikipedia article for this title in the reference index.)"
            u.context = "%s\n\n%s" % (name, text)
    finally:
        conn.close()
    return len(cache) - len(missing), missing
