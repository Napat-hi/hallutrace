# HalluTrace

Code, data and supplementary material for **HalluTrace**, which estimates where a hallucination of
an open language model most likely starts: the training data, the training process or
inference, with the knowledge boundary (the corpus holds neither fact) counted on its own. The
paper studies one error whose answer and corpus evidence can both be checked: wrong birth and
death years in biographies, judged against Wikidata. The pipeline runs eight HALoGEN tasks (3,200 prompts) on five 7B-scale models whose pretraining
corpora are searchable (OLMo 2, DCLM, OLMo 1.7, RedPajama-INCITE and Pythia), with one greedy and
five sampled answers per prompt, checks each task's verifier against human labels, and counts the
wrong and the correct fact in each model's own corpus.

Paper: AIESC 2027.

## Status

This is the release the paper cites, tagged `aiesc2027-submission`. It holds:

- the code for every stage, with its tests;
- the sampled prompt ids and seeds, and the generation plan (seeds, batches, settings);
- the generations and the verifiers' verdicts;
- the corpus counts (numbers only, no corpus text);
- the annotation guidelines, the labels and the agreement figures;
- the supplementary tables and figures, and the list of deviations from HALoGEN's released code.

HALoGEN's prompts and code are not copied here: the code runs against a pinned HALoGEN commit
with the patches in `patches/` applied, and the prompts are referenced by id.

## What is here

| Path | What it is |
|---|---|
| `data/prompt_ids.csv` | The 3,200 sampled prompts by id: task, HALoGEN source file and row. No prompt text |
| `data/prompts_manifest.json` | The draw: seed 20260916, 400 per task, strata, pool sizes, HALoGEN commit |
| `scripts/draw_sample.py` | Redraws the sample from HALoGEN's released prompt files |
| `data/generation_plan/` | One file per model: seed, sampling settings, token budget and fixed batch for each of the 96,000 generations |
| `scripts/build_generation_plan.py` | Rebuilds `data/generation_plan/` from `data/prompts.csv` |
| `modal_app.py` | The Modal app: model download, generation, verification and the Biographies run, one entrypoint per step (`modal run modal_app.py::<name>`); needs the Modal secrets named at its top |
| `patches/` | Changes to HALoGEN, applied on top of the pinned commit |
| `docs/DEVIATIONS.md` | Every change to HALoGEN's released code and why, with the generation settings |
| `docs/supplementary_methods.md` | Benchmark sample, models, generation, reproducibility, verification, attribution and compute, in full |
| `results/bio_batch_check.json`, `results/bio_filter_generations.json` | The check of batched against unbatched Biographies verification, and the atoms the Biographies filter drops, by model |
| `results/bio_scale.json` | Biographies verification time and GPU use on 10 and 50 responses, unbatched, from `modal_app.py::bio_scale` |
| `locks/` | Exact package versions, locally and in each container image |
| `hallutrace/` | Shared code: the abstention detector, outcomes and the token-limit tail rule, HALoGEN's extraction prompts and parsers, one wrapper for all verifiers, the LLM client and API cache, verifier inputs, and the Biographies run and filter |
| `scripts/generation_quality.py` | Truncation and repetition in the generations |
| `scripts/halogen_response_lengths.py`, `results/halogen_response_lengths.csv`, `results/halogen_response_lengths_by_task.csv`, `results/halogen_truncation_by_model_family.csv` | Lengths of HALoGEN's released responses by model and task, and how often they end mid-sentence near their maximum length (the script needs `responses` in the sparse checkout) |
| `hallutrace/store.py`, `scripts/init_db.py` | The SQLite store, nine tables from prompts and generations to corpus counts and labels, and the script that creates it |
| `scripts/validate_generations.py` | Checks a downloaded generation run: every batch, record and pinned model revision |
| `results/generations/` | All 96,000 generations, one JSONL file per fixed batch: prompt id, model, sample, settings, the response and why it stopped. No prompt text |
| `results/verifier/generations/`, `results/verifier/generations-revised-prompts/` | The verifiers' inputs and outputs. Biographies and Historical Events come from the run with HALoGEN's extraction prompts, the other six tasks from the run with the revised ones. Each `.meta.csv` sidecar ties a row to its generation and records its outcome |
| `results/pilot_1/`, `results/pilot_2/` | The two pilot generation runs (5,760 generations, by prompt id); some tests read these files |
| `scripts/posthoc_filter.py`, `hallutrace/posthoc.py` | Filters on verifier output: extracted units the model never wrote, `NONE` units and the decomposer's framing lines are dropped, the raw output kept |
| `scripts/gold_cut.py`, `hallutrace/gold_cut.py`, `hallutrace/gold_cut_bio.py` | The annotation sample per task and the two annotators' packets, with the verifier's answer kept apart |
| `scripts/agreement.py`, `hallutrace/agreement.py` | Agreement: annotator kappa, verifier kappa or extractor accuracy against gold, confusion matrices, intervals and the adjudication file |
| `scripts/hand_decomposition_score.py`, `hallutrace/hand_decomposition_score.py` | The hand decomposition check: the decomposer's atoms against a person's, precision, recall and F1 |
| `scripts/stability.py`, `hallutrace/stability.py` | h per item over the five samples on all eight tasks, and the regime per Biographies birth and death year |
| `hallutrace/claim_key.py`, `hallutrace/relations.py`, `data/relation_vocabulary.json` | When two differently worded claims are one claim, the relation vocabulary loader, and the vocabulary itself: 30 relations with synonyms and Wikidata properties |
| `scripts/fetch_windows.py`, `hallutrace/fetch_names.py`, `hallutrace/fetch_sampling.py`, `hallutrace/infinigram.py` | The infini-gram search: query strings from prompt names, which occurrences to download, and the rate-limited client with its window cache (corpus text, kept local and never released) |
| `hallutrace/fact_match.py`, `hallutrace/pilot_counts.py`, `scripts/pilot_counts.py`, `scripts/pilot_rule_check.py` | Year matching in a window, counts of correct birth and death years near a name, and the check of the counting rule |
| `results/counting_pilot/counting_rule_packet.csv` | The 60 hand-labelled corpus matches the counting rule was checked on (41 correct, 15 another person, 3 another fact, 1 unclear), without the corpus text. 52 `rank` values are rounded |
| `results/counting_pilot/pilot_counts.csv`, `pilot_summary.md`, `pilot_packet_strata.csv`, `pilot_precision.md`, `pilot_rule_check.md` | The 20-name counting pilot: correct-year counts per name, fact, corpus and matcher (7,200 rows) with a summary, the strata the 60 hand-labelled matches were drawn from, the precision of those matches, and the check of the chosen counting rule (precision 0.93). Rebuilt from a clean clone of this repository |
| `scripts/build_alias_table.py`, `data/alias_table.json`, `data/alias_overrides.json`, `data/alias_build/` | Prompt names to Wikidata items: the snapshot, the hand overrides and the table built from both |
| `scripts/bio_year_claims.py`, `hallutrace/bio_years.py`, `scripts/fetch_namesake_years.py` | Biographies birth and death year claims judged against Wikidata, with the years of each name's namesakes |
| `scripts/bio_conflation.py`, `hallutrace/bio_conflation.py` | Whether wrong years come from a namesake, against two chance baselines |
| `scripts/bio_counts.py`, `hallutrace/bio_counts.py` | Biographies corpus counts per fetch shard, the merge's checks, and tau per model |
| `results/bio_year_claims.csv`, `results/bio_year_claims_summary.json` | Every Biographies birth and death year claim, with the Wikidata year and the namesake check |
| `results/bio_conflation.json`, `results/bio_conflation_candidates.csv` | The namesake test: observed matches against both chance baselines, and the candidate claims |
| `results/bio_other_person_sample.csv` | 50 wrong-year claims checked by hand for whether the year belongs to someone else |
| `results/bio_counts/` | Corpus counts per fetch shard (`shard_<i>_of_3/`, numbers only) and the merge of all three (`merged/`: the counts, tau per model and the run's checks) |
| `scripts/he_tau.py`, `hallutrace/he_tau.py`, `results/slot_types/he_tau.csv` | Historical-events tau per model, at every cutoff from the 5th to the 20th percentile |
| `scripts/slot_types.py`, `hallutrace/slot_types.py`, `scripts/nh_worst_case.py` | Types for the Biographies year slots, the exact-phrase check on unknown counts, the F5 range, and a worst case for false matches in wrong-year counts |
| `scripts/intervention_points.py`, `hallutrace/intervention_points.py` | The label per Biographies year slot (intervention point, confidence, flags), the tau sweep, and the T5 and F5 tables |
| `scripts/year_judge_packet.py` | The annotation packet that checks the Wikidata year judge |
| `scripts/supplementary_figures.py`, `hallutrace/supplementary_figures.py` | F1 to F4 with the CSV each one plots, from T1, the `agreement.json` files and the filtered verifier output; drawing needs matplotlib (the figures were drawn with 3.10.9), the CSVs do not |
| `scripts/supp_tables.py`, `results/supplementary/` | T1 (h per task and model, before and after the unit filter, with the regime counts) and T2 (every sampled generation by outcome), built from two runs of `scripts/stability.py` |
| `results/hand_decomposition/` | The hand decomposition check: precision, recall and F1 per task with 95% bootstrap intervals (`t4.csv`), and the matcher's two 30-pair stress tests |
| `results/slot_types/`, `results/intervention_points/` | Historical-events counts, the Biographies year-slot types, phrase checks and F5 range, and the label for each slot with the tau sweep. CSV files only: the run summaries and metadata record local paths and are left out |
| `annotation/` | Human labels: the verifier check (50 units per task, two annotators, adjudicated gold, agreement) and the hand decomposition check; see `annotation/README.md` |
| `docs/annotation_guidelines/` | The guidelines the annotators followed, with the rulings added at adjudication |
| `tests/`, `scripts/run_tests.py` | Tests; no API key or network needed |

Scripts write to `outputs/` (not tracked) unless told otherwise.

**HALoGEN's text is not copied.** Entity names (people, events) are kept as identifiers; prompt
wording and source text are not released. In the verifier files the columns `prompt`, `passage`,
`extract_atomic_unit_prompt`, `list` and `true_answer` are kept but left empty, so every script
still reads those columns. `list_condition` and `count` stay filled: the unit filter needs these answers
to recompute the verdicts. `scripts/draw_sample.py` rebuilds the prompts from HALoGEN's own files.

**The model text is unfiltered.** The responses in the released files are the models' output as
written, as in HALoGEN's own release, and some contain e-mail addresses or phone numbers that the
models produced.

**The shard files in `results/bio_counts/`** came from the development repository. In each
shard's `meta.json`, `counting_code_sha256` hashes `hallutrace/pilot_counts.py` and
`hallutrace/fact_match.py` as those files stood before comments were removed for this release, and the
`*_git_head` fields name commits of that repository. The counting itself did not change. A
shard re-exported from this repository carries a different hash, so `bio_counts.py merge` will
not mix it with the released shards: re-export all three shards before a merge. Re-exporting needs the
window cache, which holds corpus text and is not released, so a rerun starts with
`scripts/fetch_windows.py`.

In `results/bio_counts/`, the `role` column is `c` for counts of the person's correct (Wikidata)
year and `h` for counts of a year a model stated wrongly.

## Supplementary material

The paper's supplementary tables and figures, and where each one is in this repository. The
labels are the ones the paper's text uses; the paper prints T5 as Table I, T3 as Table II, the
pipeline figure as Fig. 1 and F5 as Fig. 2.

| Label | What it shows | File |
|---|---|---|
| T1 | h per task and model, before and after the unit filter, with usable samples and regime counts | `results/supplementary/t1_h_by_task_model.csv` |
| T2 | Every sampled generation by outcome: usable, zero units, abstention, empty, emptied, missing | `results/supplementary/t2_outcomes_by_task_model.csv` |
| T3 | Verifier validation per task: verifier kappa or extractor accuracy against gold with 95% intervals, the gate, and annotator kappa | `annotation/verifier_check/<task>/agreement.json` |
| T4 | The hand decomposition check: precision, recall and F1 of the decomposer's atoms, with 95% bootstrap intervals. The paper's 0.62 and 0.86 are the Biographies row after the unit filter (`filtered`, `containment_0.8`); the `raw` row gives precision 0.59 | `results/hand_decomposition/t4.csv` (labels in `annotation/hand_decomposition/`) |
| T5 | Biographies birth and death year slots per model, by stability regime and corpus-count type; Table I sums the five models' rows | `results/intervention_points/t5.csv` |
| F1 | h per task and model as a heatmap, each task with its gate number beside it | `results/supplementary/figures/f1_h_by_task_model.{pdf,png,csv}` |
| F2 | Units dropped by the unit filter, by reason and task | `results/supplementary/figures/f2_units_dropped.{pdf,png,csv}`, totals per task in `f2_units_dropped_by_task.csv` |
| F3 | Confusion matrices of each kappa-gated verifier and the year judge against adjudicated gold; the annotator matrices are in each `agreement.json` | `results/supplementary/figures/f3_confusion_matrices.{pdf,png,csv}` |
| F4 | Stability regime shares by task and model | `results/supplementary/figures/f4_stability_regimes.{pdf,png,csv}` |
| F5 | Intervention point shares per model, as a range where the counts leave the point open | `results/intervention_points/f5_by_model.csv` |
| Corpus counts | Correct and wrong year counts per name in each model's corpus, and tau per model | `results/bio_counts/merged/` (Biographies), `results/slot_types/he_counts.csv` and `he_tau.csv` (Historical Events) |
| Deviations | Every change to HALoGEN's released code | `docs/DEVIATIONS.md` |
| S2 check | A slot's first two wrong samples agree, against a chance baseline | `results/supplementary/s2_agreement.json` (`scripts/s2_agreement.py`) |
| Methods | Sample, models, generation, reproducibility, verification, attribution (types, regimes, the point rule and the five fragility checks behind confidence) and compute in full | `docs/supplementary_methods.md` |

**Quarantined tasks.** Each task's verifier was checked against 50 human-labelled units before
its numbers could enter the analysis. A task is accepted at kappa 0.8 (accuracy 0.9 for the
three extraction tasks), tentative at 0.667 (0.8), and quarantined below. All eight verifiers
are quarantined, so the h of each appears only here, with its gate number beside it:

| Task | Measure | Value | 95% interval |
|---|---|---|---|
| Biographies | verifier kappa | 0.15 | 0.03 to 0.28 |
| Historical Events | verifier kappa | 0.52 | 0.35 to 0.70 |
| Scientific Attribution | verifier kappa | 0.20 | 0.05 to 0.36 |
| Summarization | verifier kappa | 0.60 | 0.44 to 0.75 |
| Simplification | verifier kappa | 0.44 | 0.28 to 0.62 |
| False Presuppositions | extractor accuracy | 0.66 | 0.52 to 0.78 |
| Rationalization (Binary) | extractor accuracy | 0.70 | 0.56 to 0.81 |
| Rationalization (Numerical) | extractor accuracy | 0.58 | 0.44 to 0.71 |

The Biographies year claims behind T5 and F5 are judged by Wikidata, not by the Biographies
verifier, and that judge was checked separately: kappa 0.79 (0.64 to 0.92), tentative
(`annotation/verifier_check/biographies_years/agreement.json`).

## Setup

```
git clone --filter=blob:none --sparse https://github.com/AbhilashaRavichander/HALoGEN.git external/HALoGEN
git -C external/HALoGEN sparse-checkout set prompts verifiers scorers src
git -C external/HALoGEN checkout 8b7dd77e9602af0b4c9d1430e60f474ac1b4eaa4
git -C external/HALoGEN apply ../../patches/halogen-fixes.patch
git -C external/HALoGEN apply ../../patches/halogen-openfactscore.patch
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

## Rebuilding the tables and figures

These commands read released files only. Commands that write into `results/`, `annotation/` or
`data/` overwrite the committed files, so `git diff` shows any difference (on Windows, use
`git diff --ignore-cr-at-eol`: some scripts write CRLF line endings). The rest write to `outputs/`.

Rerunning the verifiers does not reproduce the released verdicts exactly: the API cache and the
Gemma cache are not released, and the Biographies verifier samples. Everything downstream of the
verdicts recomputes from `results/verifier/` without a verifier call, with the commands below.

```
# h per item and per-slot regimes, with and without the unit filter (inputs to T1 and T2)
python scripts/stability.py --verifier-root results/verifier/generations --v2-root results/verifier/generations-revised-prompts --out outputs/stability
python scripts/stability.py --verifier-root results/verifier/generations --v2-root results/verifier/generations-revised-prompts --outputs outputs --out outputs/stability_outputs
# T1 and T2 -> results/supplementary/
python scripts/supp_tables.py --stability outputs/stability --stability-outputs outputs/stability_outputs
# S2 check -> results/supplementary/s2_agreement.json
python scripts/s2_agreement.py --verifier-root results/verifier/generations
# F2: the unit filter on the six revised-prompt tasks, then the drops -> outputs/figs/
python scripts/posthoc_filter.py results/verifier/generations-revised-prompts/outputs --out outputs/v2_filtered
python scripts/supplementary_figures.py drops outputs/v2_filtered results/verifier/generations-revised-prompts/inputs --out outputs/figs
# F1, F3 and F4 -> outputs/figs/ (drawing needs matplotlib)
python scripts/supplementary_figures.py h results/supplementary/t1_h_by_task_model.csv --out outputs/figs
python scripts/supplementary_figures.py confusion --out outputs/figs
python scripts/supplementary_figures.py regimes results/supplementary/t1_h_by_task_model.csv --out outputs/figs
# T4 and the stress tests -> results/hand_decomposition/
python scripts/hand_decomposition_score.py rescore --root biographies=results/verifier/generations --root summarization=results/verifier/generations-revised-prompts --root simplification=results/verifier/generations-revised-prompts
python scripts/hand_decomposition_score.py t4 --root biographies=results/verifier/generations --root summarization=results/verifier/generations-revised-prompts --root simplification=results/verifier/generations-revised-prompts
# T3: agreement.json per task (for biographies_years, pass annotator_a.csv as both --a and --b)
python scripts/agreement.py compute annotation/verifier_check/<task> --a annotation/verifier_check/<task>/annotator_a.csv --b annotation/verifier_check/<task>/annotator_b.csv
# the year-judge packet -> outputs/yj/, to compare with annotation/verifier_check/biographies_years/
python scripts/year_judge_packet.py results/verifier/generations --out outputs/yj
# Biographies year claims, the namesake test and the merged corpus counts -> results/
python scripts/bio_year_claims.py results/verifier/generations
python scripts/bio_conflation.py
python scripts/bio_counts.py merge
# the generation run, and the generation plan (from data/prompts.csv, which scripts/draw_sample.py writes)
python scripts/validate_generations.py results/generations
python scripts/build_generation_plan.py
```

`bio_counts.py merge` writes this repository's commit into `merged/meta.json` as `merge_git_head`;
every other output above matches the committed file byte for byte. The slot types, the
intervention points (T5, F5), Historical Events tau and the counting pilot read the infini-gram
window and phrase caches, which hold corpus text and are not released; their outputs are in
`results/slot_types/`, `results/intervention_points/` and `results/counting_pilot/`.

## Citation and questions

`CITATION.cff` holds the citation, and GitHub shows it under "Cite this repository". Questions and
problem reports go to the issue tracker: https://github.com/Napat-hi/hallutrace/issues.

## License

Code: Apache License 2.0 (`LICENSE`). Data produced for this study (labels, derived tables): CC BY 4.0 (`LICENSE-DATA`).
Model outputs and third-party material keep the original terms; code copied from other projects
is listed in `THIRD_PARTY_NOTICES.md`.
