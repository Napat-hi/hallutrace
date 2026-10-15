# HalluTrace

Code, data and supplementary material for **HalluTrace**, which traces each hallucination of an
open language model to where it most likely starts: the training data, the training process, or
inference. It runs eight HALoGEN tasks (3,200 prompts) on five 7B-scale models whose pretraining
corpora are searchable (OLMo 2, DCLM, OLMo 1.7, RedPajama-INCITE and Pythia), with one greedy and
five sampled answers per prompt, checks each task's verifier against human labels, and counts the
wrong and the correct fact in each model's own corpus.

Paper: AIESC 2027 (link to follow).

## Status

This repository is being filled in until 19 October 2026. When it is complete it will hold:

- the code for every stage, with its tests;
- the sampled prompt ids and seeds, and the generation plan (seeds, batches, settings);
- the generations and the verifiers' verdicts;
- the corpus counts (numbers only, no corpus text);
- the annotation guidelines, the labels and the agreement figures;
- the supplementary tables and figures, and the list of deviations from HALoGEN's released code.

HALoGEN's prompts and code are not copied here: the code runs against a pinned HALoGEN commit
with our patches applied, and the prompts are referenced by id.

## What is here so far

| Path | What it is |
|---|---|
| `data/prompt_ids.csv` | The 3,200 sampled prompts by id: task, HALoGEN source file and row. No prompt text |
| `data/prompts_manifest.json` | The draw: seed 20260916, 400 per task, strata, pool sizes, HALoGEN commit |
| `scripts/draw_sample.py` | Redraws the sample from HALoGEN's released prompt files |
| `data/generation_plan/` | One file per model: seed, sampling settings, token budget and fixed batch for each of the 96,000 generations |
| `scripts/build_generation_plan.py` | Rebuilds `data/generation_plan/` from `data/prompts.csv` |
| `modal_app.py` | The Modal app: model download, generation, verification and the Biographies run, one entrypoint per step (`modal run modal_app.py::<name>`); needs the Modal secrets named at its top |
| `patches/` | Our changes to HALoGEN, applied on top of the pinned commit |
| `locks/` | Exact package versions, locally and in each container image |
| `hallutrace/` | Shared code: the abstention detector, outcomes and the token-limit tail rule, HALoGEN's extraction prompts and parsers, one wrapper for all verifiers, the LLM client and API cache, verifier inputs, and the Biographies run and filter |
| `scripts/generation_quality.py` | Truncation and repetition in the generations |
| `scripts/halogen_response_lengths.py` | Lengths of HALoGEN's released responses (needs `responses` in the sparse checkout) |
| `hallutrace/store.py`, `scripts/init_db.py` | The SQLite store, nine tables from prompts and generations to corpus counts and labels, and the script that creates it |
| `scripts/validate_a2.py` | Checks a downloaded generation run: every batch, record and pinned model revision |
| `results/pilot/`, `results/pilot2/` | The two pilot generation runs (5,760 generations, by prompt id); some tests read them |
| `scripts/posthoc_filter.py`, `hallutrace/posthoc.py` | Filters on verifier output: extracted units the model never wrote, `NONE` units and the decomposer's framing lines are dropped, the raw output kept |
| `scripts/gold_cut.py`, `hallutrace/gold_cut.py`, `hallutrace/gold_cut_bio.py` | The annotation sample per task and the two annotators' packets, with the verifier's answer kept apart |
| `scripts/agreement.py`, `hallutrace/agreement.py` | Agreement: annotator kappa, verifier kappa or extractor accuracy against gold, confusion matrices, intervals and the adjudication file |
| `scripts/e3_score.py`, `hallutrace/e3_score.py` | The hand decomposition check: the decomposer's atoms against a person's, precision, recall and F1 |
| `scripts/stability.py`, `hallutrace/stability.py` | h per item over the five samples on all eight tasks, and the regime per Biographies birth and death year |
| `hallutrace/claim_key.py`, `hallutrace/relations.py` | When two differently worded claims are one claim, and the relation vocabulary loader |
| `scripts/fetch_windows.py`, `hallutrace/fetch_names.py`, `hallutrace/fetch_sampling.py`, `hallutrace/infinigram.py` | The infini-gram search: query strings from prompt names, which occurrences to download, and the rate-limited client with its window cache (corpus text, kept local and never released) |
| `hallutrace/fact_match.py`, `hallutrace/pilot_counts.py`, `scripts/pilot_counts.py`, `scripts/pilot_rule_check.py` | Year matching in a window, counts of correct birth and death years near a name, and the check of the counting rule |
| `scripts/build_alias_table.py`, `data/alias_table.json`, `data/alias_overrides.json`, `data/alias_build/` | Prompt names to Wikidata items: the snapshot, the hand overrides and the table built from them |
| `scripts/bio_year_claims.py`, `hallutrace/bio_years.py`, `scripts/fetch_namesake_years.py` | Biographies birth and death year claims judged against Wikidata, with the years of each name's namesakes |
| `scripts/bio_conflation.py`, `hallutrace/bio_conflation.py` | Whether wrong years come from a namesake, against two chance baselines |
| `scripts/bio_counts.py`, `hallutrace/bio_counts.py` | Biographies corpus counts per fetch shard, the merge's checks, and tau per model |
| `results/bio_year_claims.csv`, `results/bio_year_claims_summary.json` | Every Biographies birth and death year claim, with the Wikidata year and the namesake check |
| `results/bio_conflation.json`, `results/bio_conflation_candidates.csv` | The namesake test: observed matches against both chance baselines, and the candidate claims |
| `results/bio_other_person_sample.csv` | 50 wrong-year claims checked by hand for whether the year belongs to someone else |
| `results/bio_counts/` | Corpus counts per fetch shard (`shard_<i>_of_3/`, numbers only) and their merge (`merged/`: the counts, tau per model and the run's checks) |
| `scripts/c_exec_tau.py`, `hallutrace/c_exec_tau.py`, `results/c_exec/he_tau.csv` | Historical-events tau per model, at every cutoff from the 5th to the 20th percentile |
| `scripts/c_exec_untyped.py`, `hallutrace/c_exec_untyped.py`, `scripts/c_exec_nh_sensitivity.py` | Types for the Biographies year slots, the exact-phrase check on unknown counts, the F5 range, and a worst case for false matches in wrong-year counts |
| `scripts/c_exec2.py`, `hallutrace/c_exec2.py` | The label per Biographies year slot (intervention point, confidence, flags), the tau sweep, and the T5 and F5 tables |
| `scripts/year_judge_packet.py` | The annotation packet that checks the Wikidata year judge |
| `scripts/supp_tables.py`, `results/supplementary/` | T1 (h per task and model, before and after the unit filter, with the regime counts) and T2 (every sampled generation by outcome), built from two runs of `scripts/stability.py` |
| `results/e3/` | The hand decomposition check: precision, recall and F1 per task with 95% bootstrap intervals (`t4.csv`), and the matcher's two 30-pair stress tests |
| `results/c_exec/`, `results/c_exec2/` | Historical-events counts, the Biographies year-slot types, phrase checks and F5 range, and the label for each slot with the tau sweep. CSV files only for now; the summaries and run metadata follow |
| `annotation/` | Human labels: the verifier check (50 units per task, two annotators, adjudicated gold, agreement) and the hand decomposition check; see `annotation/README.md` |
| `docs/annotation_guidelines/` | The guidelines the annotators followed |
| `tests/`, `scripts/run_tests.py` | Tests; no API key or network needed |

Scripts write to `outputs/` (not tracked) unless told otherwise.

**The shard files in `results/bio_counts/`** came from our development repository. In each
shard's `meta.json`, `counting_code_sha256` hashes `hallutrace/pilot_counts.py` and
`hallutrace/fact_match.py` as they stood before we removed comments for this release, and the
`*_git_head` fields name commits of that repository. The counting itself did not change. A
shard re-exported from this repository carries a different hash, so `bio_counts.py merge` will
not mix it with ours: re-export all three shards to merge your own. Re-exporting needs the
window cache, which holds corpus text and is not released, so a rerun starts with
`scripts/fetch_windows.py`.

## Setup

```
git clone --filter=blob:none --sparse https://github.com/AbhilashaRavichander/HALoGEN.git external/HALoGEN
git -C external/HALoGEN sparse-checkout set prompts verifiers scorers src
git -C external/HALoGEN checkout 8b7dd77e9602af0b4c9d1430e60f474ac1b4eaa4
git -C external/HALoGEN apply ../../patches/halogen-p0-7.patch
git -C external/HALoGEN apply ../../patches/halogen-p1-8-openfactscore.patch
python -m pip install -r locks/local.txt
```

The patched HALoGEN imports `hallutrace`, so the repository root must be on `PYTHONPATH`.

## Reproducing the prompt sample

```
python scripts/draw_sample.py
```

This writes `data/prompts.csv` with the prompt text, locally. Its first four columns match
`data/prompt_ids.csv` row for row, and it repairs 15 historical-events names that the
released file stores double-encoded.

## Tests

```
python scripts/draw_sample.py   # some tests read data/prompts.csv
python scripts/run_tests.py
```

## License

Code: Apache License 2.0 (`LICENSE`). Data we produced (labels, derived tables): CC BY 4.0.
Model outputs and third-party material keep their own terms; code copied from other projects
is listed in `THIRD_PARTY_NOTICES.md`.
