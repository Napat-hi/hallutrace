# Counting rule check

Numbers only. Written by `scripts/pilot_rule_check.py` from the labelled packet (60 of 60 rows labelled), `pilot_packet_strata.csv` and the window cache (read-only). Short forms of disambiguated names (source `stripped`) are left out throughout: they were correct in 1 of 9 scored packet rows (`pilot_precision.md`).

## 1. Precision on the packet

Per stratum (nearest occurrence's distance band x relation cue), non-stripped rows. Strata with fewer than 3 scored rows use the pooled precision of their cue value.

| band | cue | correct of scored | precision used | candidates in population |
|---|---|---|---|---|
| 0-100 | 0 | 12 of 14 | 0.86 | 631 |
| 0-100 | 1 | 5 of 5 | 1.00 | 332 |
| 101-200 | 0 | 2 of 5 | 0.40 | 104 |
| 101-200 | 1 | 4 of 4 | 1.00 | 9 |
| 201-400 | 0 | 2 of 5 | 0.40 | 58 |
| 201-400 | 1 | 4 of 4 | 1.00 | 166 |
| 401-800 | 0 | 2 of 2 | 0.64 | 105 |
| 401-800 | 1 | 4 of 4 | 1.00 | 38 |
| 801-1600 | 0 | 0 of 2 | 0.64 | 141 |
| 801-1600 | 1 | 5 of 5 | 1.00 | 86 |

### As labelled (50 scored rows)

| rule | weighted precision | kept | raw correct | 95% Wilson interval (raw) |
|---|---|---|---|---|
| cue, whole window | 1.00 | 0.45 | 22 of 22 | 0.85 to 1.00 |
| bare, whole window | 0.84 | 1.00 | 40 of 50 | 0.67 to 0.89 |
| bare <= 100 | 0.91 | 0.63 | 17 of 19 | 0.69 to 0.97 |
| bare <= 100 OR cue (chosen) | 0.93 | 0.84 | 34 of 36 | 0.82 to 0.98 |

### Identical snippets counted once (57 of 60 rows kept)

| rule | weighted precision | kept | raw correct | 95% Wilson interval (raw) |
|---|---|---|---|---|
| cue, whole window | 1.00 | 0.45 | 20 of 20 | 0.84 to 1.00 |
| bare, whole window | 0.83 | 1.00 | 37 of 47 | 0.65 to 0.88 |
| bare <= 100 | 0.91 | 0.63 | 17 of 19 | 0.69 to 0.97 |
| bare <= 100 OR cue (chosen) | 0.93 | 0.84 | 32 of 34 | 0.81 to 0.98 |

### Without Aleksei Musin-Pushkin (15 of 60 rows)

| rule | weighted precision | kept | raw correct | 95% Wilson interval (raw) |
|---|---|---|---|---|
| cue, whole window | 1.00 | 0.46 | 16 of 16 | 0.81 to 1.00 |
| bare, whole window | 0.82 | 1.00 | 27 of 35 | 0.61 to 0.88 |
| bare <= 100 | 0.91 | 0.64 | 11 of 12 | 0.65 to 0.99 |
| bare <= 100 OR cue (chosen) | 0.93 | 0.86 | 22 of 23 | 0.79 to 0.99 |

### P019 and P044 relabelled correct

| rule | weighted precision | kept | raw correct | 95% Wilson interval (raw) |
|---|---|---|---|---|
| cue, whole window | 1.00 | 0.45 | 22 of 22 | 0.85 to 1.00 |
| bare, whole window | 0.85 | 1.00 | 41 of 50 | 0.69 to 0.90 |
| bare <= 100 | 0.91 | 0.62 | 17 of 19 | 0.69 to 0.97 |
| bare <= 100 OR cue (chosen) | 0.93 | 0.83 | 34 of 36 | 0.82 to 0.98 |

## 2. n_c on the 20 pilot names

Distinct documents per (name, fact, index) under each rule; facts whose strings are all fully fetched (docs units) only. p10 is nearest-rank over n_c > 0. A 20-name preview, not tau.

| rule | index | facts | n_c = 0 | n_c > 0 | p10 (n_c > 0) | median (incl. 0) |
|---|---|---|---|---|---|---|
| cue, whole window | OLMo 2 | 30 | 16 | 14 | 1 | 0 |
| cue, whole window | DCLM | 30 | 21 | 9 | 1 | 0 |
| cue, whole window | Dolma | 32 | 13 | 19 | 1 | 1 |
| cue, whole window | RedPajama | 28 | 6 | 22 | 1 | 3 |
| cue, whole window | Pile | 34 | 9 | 25 | 3 | 3 |
| bare, whole window | OLMo 2 | 30 | 7 | 23 | 1 | 1 |
| bare, whole window | DCLM | 30 | 17 | 13 | 1 | 0 |
| bare, whole window | Dolma | 32 | 5 | 27 | 1 | 2 |
| bare, whole window | RedPajama | 28 | 3 | 25 | 1 | 5 |
| bare, whole window | Pile | 34 | 2 | 32 | 3 | 6 |
| bare <= 100 | OLMo 2 | 30 | 7 | 23 | 1 | 1 |
| bare <= 100 | DCLM | 30 | 17 | 13 | 1 | 0 |
| bare <= 100 | Dolma | 32 | 5 | 27 | 1 | 2 |
| bare <= 100 | RedPajama | 28 | 3 | 25 | 1 | 4 |
| bare <= 100 | Pile | 34 | 4 | 30 | 3 | 5 |
| bare <= 100 OR cue (chosen) | OLMo 2 | 30 | 7 | 23 | 1 | 1 |
| bare <= 100 OR cue (chosen) | DCLM | 30 | 17 | 13 | 1 | 0 |
| bare <= 100 OR cue (chosen) | Dolma | 32 | 5 | 27 | 1 | 2 |
| bare <= 100 OR cue (chosen) | RedPajama | 28 | 3 | 25 | 1 | 4 |
| bare <= 100 OR cue (chosen) | Pile | 34 | 4 | 30 | 3 | 6 |

## 3. Names counted without their short form

38 of the 444 names in the cache have a short (stripped) form; with it left out, their n_c comes from the name as written and any aliases, so it is a lower bound.
