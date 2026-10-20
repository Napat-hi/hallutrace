# Annotation

Human labels used to check the verifiers (`verifier_check/`) and the decomposer (`hand_decomposition/`). The guidelines the
annotators followed are in `docs/annotation_guidelines/`.

## `verifier_check/<task>/`: verifier check, 50 units per task

Each task's 50 units were drawn from the verifier output with a fixed seed (in `manifest.json`),
stratified so that units the verifier flagged make up 38% to 58% of the sample. Two annotators
labelled every unit independently without seeing the verifier's answer, and a third team member
settled the disagreements.

| File | What it is |
|---|---|
| `annotator.csv` | The blank sheet both annotators received |
| `annotator_a.csv`, `annotator_b.csv` | The two annotators' labels (`annotator_label`), tags and notes |
| `adjudication.csv` | The disagreements, each annotator's label and notes, and the settled `gold_label` with its reason |
| `key.csv` | The verifier's answer for each unit and where the unit comes from (`generation_id`, `model`, `source_file`) |
| `manifest.json` | The draw: seed, pool, flagged share, the question asked |
| `agreement.json` | Annotator kappa, the verifier's kappa (or the extractor's accuracy) against gold, confusion matrices and 95% intervals, with and without units tagged `not a claim` |

`context_shown` is left blank for Biographies, Summarization and Simplification. There it held
third-party text: the Wikipedia article for Biographies, and HALoGEN's source passage for the
other two. The article is found by the person's name, and the passage by the prompt id in
`data/prompt_ids.csv`. The other tasks show the model's own response, which is kept.

`agreement.json` is rebuilt with:

```
python scripts/agreement.py compute annotation/verifier_check/<task> --a annotation/verifier_check/<task>/annotator_a.csv --b annotation/verifier_check/<task>/annotator_b.csv
```

`verifier_check/biographies_years/` checks a different judge: the exact match of a stated birth or death
year against Wikidata. It has one annotator, so it is scored with `annotator_a.csv` as both A
and B, and its own `README.md` gives the instructions. `scripts/year_judge_packet.py` draws it.

## `hand_decomposition/<task>/`: hand decomposition, 30 generations per task

For Biographies, Summarization and Simplification, 30 generations per task (6 per model, seed in
`manifest.json`) were drawn by `hallutrace/decomp_cut.py`. One annotator split each response
into atomic facts, one per line in `annotator.csv`'s `human_atoms`, without seeing the
decomposer's atoms. `key.csv` joins each item to its generation. `scripts/hand_decomposition_score.py` scores
the decomposer's atoms against these.
