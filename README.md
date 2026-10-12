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
| `tests/`, `scripts/run_tests.py` | Tests; no API key or network needed |

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
