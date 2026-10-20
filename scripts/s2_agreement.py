"""Is S2 more than chance? First-two-wrong-sample agreement on the Biographies year slots.

    python scripts/s2_agreement.py --verifier-root results/verifier/generations

For every S2 and S3 slot (the regimes from `hallutrace.stability.slot_items`), the slot's
first two wrong samples, in sample order, agree when they state the same wrong years. The
chance baseline pairs the first wrong sample of each S2 or S3 slot with the second wrong
sample of every other S2 or S3 slot of the same model and kind (birth or death). The year claims are
results/bio_year_claims.csv, kept to generations whose outcome is `ok`, as in
`scripts/stability.py`. Writes results/supplementary/s2_agreement.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hallutrace import stability as st  # noqa: E402


def wrong_sequences(claims):
    """(model, prompt_id, kind) -> the wrong-year tuples of the slot's wrong samples, in
    sample order, for slots with Wikidata years."""
    groups = defaultdict(lambda: defaultdict(list))
    for c in claims:
        if not c["wikidata_years"].strip():
            continue
        pid = c["generation_id"].split("|")[1]
        groups[(c["model"], pid, c["kind"])][int(c["sample_index"])].append(c)
    out = {}
    for key, by_index in groups.items():
        states = [st._slot_state(by_index.get(i, [])) for i in st.SAMPLED]
        out[key] = [w for s, w in states if s == "wrong"]
    return out


def agreement(claims):
    slots = st.slot_items(claims)
    regime = {(r["model"], r["prompt_id"], r["kind"]): r["regime"] for r in slots}
    seqs = wrong_sequences(claims)
    s23 = sorted(k for k, r in regime.items() if r in ("S2", "S3"))
    agree = sum(seqs[k][0] == seqs[k][1] for k in s23)
    by_group = defaultdict(list)
    for k in s23:
        by_group[(k[0], k[2])].append(k)
    pairs = hits = 0
    for keys in by_group.values():
        for a in keys:
            for b in keys:
                if a == b:
                    continue
                pairs += 1
                hits += seqs[a][0] == seqs[b][1]
    return {
        "s2_s3_slots": len(s23),
        "first_two_wrong_agree": agree,
        "first_two_wrong_agree_share": round(agree / len(s23), 4),
        "baseline_pairs": pairs,
        "baseline_pairs_agree": hits,
        "baseline_share": round(hits / pairs, 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verifier-root", required=True)
    ap.add_argument("--claims", default=os.path.join(ROOT, "results", "bio_year_claims.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "supplementary", "s2_agreement.json"))
    args = ap.parse_args(argv)
    samples = st.bio_samples(args.verifier_root)
    with open(args.claims, encoding="utf-8", newline="") as f:
        claims, _ = st.ok_claims(csv.DictReader(f), samples)
    result = agreement(claims)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
