# Biographies year judge: gold check

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
  the other-person check (Limitations section 6).

A year the response copies from the prompt (for example a name such as "Felipe (footballer,
born 1977)") still counts as stated by the response.

Copy `annotator.csv` to `annotator_a.csv`, label it, commit it. Score, with your file as
both A and B (one annotator, so gold is your label and the inter-annotator line means nothing):

    .\.venv\Scripts\python.exe scripts\agreement.py compute annotation\e1\biographies_years --a annotation\e1\biographies_years\annotator_a.csv --b annotation\e1\biographies_years\annotator_a.csv

The gate is the verifier kappa on all rows: accept at 0.800, tentative from 0.667,
quarantine below. At 0.667 or above T5's Biographies columns and F5 stay in the body; below,
they go to the supplementary material.
