# Pilot preview over 20 names, NOT tau

Nothing here sets tau or picks the counting rule. Every number below is computed from the rows of pilot_counts.csv (7200 rows, 20 names, 40 name-fact-year facts, 5 indexes).

Reading guide: a fact is one (name, fact, year). `counted` = at least one window matched; `zero` = every string fully fetched and nothing matched (a true 0); `unknown` = nothing matched but a string was sampled or incomplete (never a 0, kept out of tau). Percentiles use the nearest-rank method (sorted value at rank ceil(p/100 x n); the median is the 50th percentile, the lower middle value for an even n) and cover facts with unit `docs` only. The `incl. 0` columns include true zeros. `facts n_c>0` and `p10 n_c>0` follow tau's population (facts with n_c > 0) restricted to `docs` units: they are still a 20-name preview, NOT tau. Famous names have sampled pools, so their n_c is in `est_mentions` units and appears only in the `est_mentions facts` and `median est_mentions` columns, never mixed with `docs` values.

## Per index and rule

| index | strings | matcher | width | counted | zero | unknown | docs facts | median n_c (incl. 0) | p10 n_c (incl. 0) | facts n_c>0 | p10 n_c>0 | est_mentions facts | median est_mentions |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DCLM | all | bare | 100 | 19 | 9 | 12 | 18 | 0 | 0 | 9 | 1 | 10 | 21 |
| DCLM | all | bare | 200 | 19 | 9 | 12 | 18 | 0 | 0 | 9 | 1 | 10 | 21 |
| DCLM | all | bare | 400 | 19 | 9 | 12 | 18 | 0 | 0 | 9 | 1 | 10 | 21 |
| DCLM | all | bare | 800 | 22 | 9 | 9 | 18 | 0 | 0 | 9 | 1 | 13 | 1444.4 |
| DCLM | all | bare | 1600 | 23 | 9 | 8 | 18 | 0 | 0 | 9 | 1 | 14 | 624 |
| DCLM | all | bare | full | 23 | 9 | 8 | 18 | 0 | 0 | 9 | 1 | 14 | 624 |
| DCLM | all | cue | 100 | 13 | 11 | 16 | 18 | 0 | 0 | 7 | 1 | 6 | 10119.9 |
| DCLM | all | cue | 200 | 14 | 11 | 15 | 18 | 0 | 0 | 7 | 1 | 7 | 10119.9 |
| DCLM | all | cue | 400 | 14 | 11 | 15 | 18 | 0 | 0 | 7 | 1 | 7 | 10119.9 |
| DCLM | all | cue | 800 | 14 | 11 | 15 | 18 | 0 | 0 | 7 | 1 | 7 | 10119.9 |
| DCLM | all | cue | 1600 | 14 | 11 | 15 | 18 | 0 | 0 | 7 | 1 | 7 | 10119.9 |
| DCLM | all | cue | full | 14 | 11 | 15 | 18 | 0 | 0 | 7 | 1 | 7 | 10119.9 |
| DCLM | all | near_or_cue | 100 | 19 | 9 | 12 | 18 | 0 | 0 | 9 | 1 | 10 | 21 |
| DCLM | all | near_or_cue | 200 | 19 | 9 | 12 | 18 | 0 | 0 | 9 | 1 | 10 | 21 |
| DCLM | all | near_or_cue | 400 | 19 | 9 | 12 | 18 | 0 | 0 | 9 | 1 | 10 | 21 |
| DCLM | all | near_or_cue | 800 | 19 | 9 | 12 | 18 | 0 | 0 | 9 | 1 | 10 | 21 |
| DCLM | all | near_or_cue | 1600 | 19 | 9 | 12 | 18 | 0 | 0 | 9 | 1 | 10 | 21 |
| DCLM | all | near_or_cue | full | 19 | 9 | 12 | 18 | 0 | 0 | 9 | 1 | 10 | 21 |
| DCLM | no_stripped | bare | 100 | 19 | 17 | 4 | 30 | 0 | 0 | 13 | 1 | 6 | 10119.9 |
| DCLM | no_stripped | bare | 200 | 19 | 17 | 4 | 30 | 0 | 0 | 13 | 1 | 6 | 10119.9 |
| DCLM | no_stripped | bare | 400 | 19 | 17 | 4 | 30 | 0 | 0 | 13 | 1 | 6 | 10119.9 |
| DCLM | no_stripped | bare | 800 | 22 | 17 | 1 | 30 | 0 | 0 | 13 | 1 | 9 | 26237.8 |
| DCLM | no_stripped | bare | 1600 | 22 | 17 | 1 | 30 | 0 | 0 | 13 | 1 | 9 | 26237.8 |
| DCLM | no_stripped | bare | full | 22 | 17 | 1 | 30 | 0 | 0 | 13 | 1 | 9 | 26237.8 |
| DCLM | no_stripped | cue | 100 | 13 | 21 | 6 | 30 | 0 | 0 | 9 | 1 | 4 | 10119.9 |
| DCLM | no_stripped | cue | 200 | 14 | 21 | 5 | 30 | 0 | 0 | 9 | 1 | 5 | 10119.9 |
| DCLM | no_stripped | cue | 400 | 14 | 21 | 5 | 30 | 0 | 0 | 9 | 1 | 5 | 14248.1 |
| DCLM | no_stripped | cue | 800 | 14 | 21 | 5 | 30 | 0 | 0 | 9 | 1 | 5 | 20239.9 |
| DCLM | no_stripped | cue | 1600 | 14 | 21 | 5 | 30 | 0 | 0 | 9 | 1 | 5 | 20239.9 |
| DCLM | no_stripped | cue | full | 14 | 21 | 5 | 30 | 0 | 0 | 9 | 1 | 5 | 20239.9 |
| DCLM | no_stripped | near_or_cue | 100 | 19 | 17 | 4 | 30 | 0 | 0 | 13 | 1 | 6 | 10119.9 |
| DCLM | no_stripped | near_or_cue | 200 | 19 | 17 | 4 | 30 | 0 | 0 | 13 | 1 | 6 | 10119.9 |
| DCLM | no_stripped | near_or_cue | 400 | 19 | 17 | 4 | 30 | 0 | 0 | 13 | 1 | 6 | 10119.9 |
| DCLM | no_stripped | near_or_cue | 800 | 19 | 17 | 4 | 30 | 0 | 0 | 13 | 1 | 6 | 10119.9 |
| DCLM | no_stripped | near_or_cue | 1600 | 19 | 17 | 4 | 30 | 0 | 0 | 13 | 1 | 6 | 10119.9 |
| DCLM | no_stripped | near_or_cue | full | 19 | 17 | 4 | 30 | 0 | 0 | 13 | 1 | 6 | 10119.9 |
| Dolma | all | bare | 100 | 30 | 1 | 9 | 20 | 3 | 1 | 19 | 1 | 11 | 2 |
| Dolma | all | bare | 200 | 32 | 1 | 7 | 20 | 3 | 1 | 19 | 1 | 13 | 2 |
| Dolma | all | bare | 400 | 32 | 1 | 7 | 20 | 3 | 1 | 19 | 1 | 13 | 34 |
| Dolma | all | bare | 800 | 34 | 1 | 5 | 20 | 3 | 1 | 19 | 1 | 15 | 35.2 |
| Dolma | all | bare | 1600 | 34 | 1 | 5 | 20 | 3 | 1 | 19 | 1 | 15 | 67.5 |
| Dolma | all | bare | full | 34 | 1 | 5 | 20 | 3 | 1 | 19 | 1 | 15 | 67.5 |
| Dolma | all | cue | 100 | 17 | 9 | 14 | 20 | 1 | 0 | 11 | 1 | 6 | 2 |
| Dolma | all | cue | 200 | 19 | 8 | 13 | 20 | 1 | 0 | 12 | 1 | 7 | 2 |
| Dolma | all | cue | 400 | 19 | 8 | 13 | 20 | 1 | 0 | 12 | 1 | 7 | 32.2 |
| Dolma | all | cue | 800 | 22 | 8 | 10 | 20 | 1 | 0 | 12 | 1 | 10 | 32.2 |
| Dolma | all | cue | 1600 | 25 | 6 | 9 | 20 | 1 | 0 | 14 | 1 | 11 | 34 |
| Dolma | all | cue | full | 25 | 6 | 9 | 20 | 1 | 0 | 14 | 1 | 11 | 34 |
| Dolma | all | near_or_cue | 100 | 30 | 1 | 9 | 20 | 3 | 1 | 19 | 1 | 11 | 2 |
| Dolma | all | near_or_cue | 200 | 30 | 1 | 9 | 20 | 3 | 1 | 19 | 1 | 11 | 2 |
| Dolma | all | near_or_cue | 400 | 30 | 1 | 9 | 20 | 3 | 1 | 19 | 1 | 11 | 2 |
| Dolma | all | near_or_cue | 800 | 32 | 1 | 7 | 20 | 3 | 1 | 19 | 1 | 13 | 34 |
| Dolma | all | near_or_cue | 1600 | 33 | 1 | 6 | 20 | 3 | 1 | 19 | 1 | 14 | 34 |
| Dolma | all | near_or_cue | full | 33 | 1 | 6 | 20 | 3 | 1 | 19 | 1 | 14 | 34 |
| Dolma | no_stripped | bare | 100 | 29 | 5 | 6 | 32 | 2 | 0 | 27 | 1 | 2 | 8394.7 |
| Dolma | no_stripped | bare | 200 | 31 | 5 | 4 | 32 | 2 | 0 | 27 | 1 | 4 | 16789.4 |
| Dolma | no_stripped | bare | 400 | 31 | 5 | 4 | 32 | 2 | 0 | 27 | 1 | 4 | 25184 |
| Dolma | no_stripped | bare | 800 | 32 | 5 | 3 | 32 | 2 | 0 | 27 | 1 | 5 | 25184 |
| Dolma | no_stripped | bare | 1600 | 32 | 5 | 3 | 32 | 2 | 0 | 27 | 1 | 5 | 25184 |
| Dolma | no_stripped | bare | full | 32 | 5 | 3 | 32 | 2 | 0 | 27 | 1 | 5 | 25184 |
| Dolma | no_stripped | cue | 100 | 17 | 17 | 6 | 32 | 0 | 0 | 15 | 1 | 2 | 8394.7 |
| Dolma | no_stripped | cue | 200 | 18 | 16 | 6 | 32 | 0 | 0 | 16 | 1 | 2 | 8394.7 |
| Dolma | no_stripped | cue | 400 | 18 | 16 | 6 | 32 | 0 | 0 | 16 | 1 | 2 | 12592 |
| Dolma | no_stripped | cue | 800 | 21 | 15 | 4 | 32 | 1 | 0 | 17 | 1 | 4 | 12592 |
| Dolma | no_stripped | cue | 1600 | 24 | 13 | 3 | 32 | 1 | 0 | 19 | 1 | 5 | 12592 |
| Dolma | no_stripped | cue | full | 24 | 13 | 3 | 32 | 1 | 0 | 19 | 1 | 5 | 12592 |
| Dolma | no_stripped | near_or_cue | 100 | 29 | 5 | 6 | 32 | 2 | 0 | 27 | 1 | 2 | 8394.7 |
| Dolma | no_stripped | near_or_cue | 200 | 29 | 5 | 6 | 32 | 2 | 0 | 27 | 1 | 2 | 12592 |
| Dolma | no_stripped | near_or_cue | 400 | 29 | 5 | 6 | 32 | 2 | 0 | 27 | 1 | 2 | 16789.4 |
| Dolma | no_stripped | near_or_cue | 800 | 31 | 5 | 4 | 32 | 2 | 0 | 27 | 1 | 4 | 16789.4 |
| Dolma | no_stripped | near_or_cue | 1600 | 32 | 5 | 3 | 32 | 2 | 0 | 27 | 1 | 5 | 16789.4 |
| Dolma | no_stripped | near_or_cue | full | 32 | 5 | 3 | 32 | 2 | 0 | 27 | 1 | 5 | 16789.4 |
| OLMo 2 | all | bare | 100 | 27 | 3 | 10 | 18 | 2 | 0 | 15 | 1 | 12 | 1 |
| OLMo 2 | all | bare | 200 | 29 | 3 | 8 | 18 | 2 | 0 | 15 | 1 | 14 | 5.2 |
| OLMo 2 | all | bare | 400 | 31 | 3 | 6 | 18 | 2 | 0 | 15 | 1 | 16 | 5.2 |
| OLMo 2 | all | bare | 800 | 33 | 3 | 4 | 18 | 2 | 0 | 15 | 1 | 18 | 21.9 |
| OLMo 2 | all | bare | 1600 | 34 | 3 | 3 | 18 | 2 | 0 | 15 | 1 | 19 | 54.1 |
| OLMo 2 | all | bare | full | 34 | 3 | 3 | 18 | 2 | 0 | 15 | 1 | 19 | 54.1 |
| OLMo 2 | all | cue | 100 | 13 | 9 | 18 | 18 | 0 | 0 | 9 | 1 | 4 | 1 |
| OLMo 2 | all | cue | 200 | 15 | 9 | 16 | 18 | 0 | 0 | 9 | 1 | 6 | 5.2 |
| OLMo 2 | all | cue | 400 | 17 | 9 | 14 | 18 | 0 | 0 | 9 | 1 | 8 | 5.2 |
| OLMo 2 | all | cue | 800 | 18 | 9 | 13 | 18 | 0 | 0 | 9 | 1 | 9 | 5.2 |
| OLMo 2 | all | cue | 1600 | 19 | 8 | 13 | 18 | 1 | 0 | 10 | 1 | 9 | 5.2 |
| OLMo 2 | all | cue | full | 19 | 8 | 13 | 18 | 1 | 0 | 10 | 1 | 9 | 5.2 |
| OLMo 2 | all | near_or_cue | 100 | 27 | 3 | 10 | 18 | 2 | 0 | 15 | 1 | 12 | 1 |
| OLMo 2 | all | near_or_cue | 200 | 29 | 3 | 8 | 18 | 2 | 0 | 15 | 1 | 14 | 5.2 |
| OLMo 2 | all | near_or_cue | 400 | 29 | 3 | 8 | 18 | 2 | 0 | 15 | 1 | 14 | 5.2 |
| OLMo 2 | all | near_or_cue | 800 | 29 | 3 | 8 | 18 | 2 | 0 | 15 | 1 | 14 | 5.2 |
| OLMo 2 | all | near_or_cue | 1600 | 29 | 3 | 8 | 18 | 2 | 0 | 15 | 1 | 14 | 5.2 |
| OLMo 2 | all | near_or_cue | full | 29 | 3 | 8 | 18 | 2 | 0 | 15 | 1 | 14 | 5.2 |
| OLMo 2 | no_stripped | bare | 100 | 26 | 7 | 7 | 30 | 1 | 0 | 23 | 1 | 3 | 28568.2 |
| OLMo 2 | no_stripped | bare | 200 | 27 | 7 | 6 | 30 | 1 | 0 | 23 | 1 | 4 | 28568.2 |
| OLMo 2 | no_stripped | bare | 400 | 29 | 7 | 4 | 30 | 1 | 0 | 23 | 1 | 6 | 26321.6 |
| OLMo 2 | no_stripped | bare | 800 | 31 | 7 | 2 | 30 | 1 | 0 | 23 | 1 | 8 | 10144.9 |
| OLMo 2 | no_stripped | bare | 1600 | 31 | 7 | 2 | 30 | 1 | 0 | 23 | 1 | 8 | 10144.9 |
| OLMo 2 | no_stripped | bare | full | 31 | 7 | 2 | 30 | 1 | 0 | 23 | 1 | 8 | 10144.9 |
| OLMo 2 | no_stripped | cue | 100 | 13 | 19 | 8 | 30 | 0 | 0 | 11 | 1 | 2 | 14284.1 |
| OLMo 2 | no_stripped | cue | 200 | 14 | 19 | 7 | 30 | 0 | 0 | 11 | 1 | 3 | 28568.2 |
| OLMo 2 | no_stripped | cue | 400 | 16 | 18 | 6 | 30 | 0 | 0 | 12 | 1 | 4 | 10144.9 |
| OLMo 2 | no_stripped | cue | 800 | 17 | 17 | 6 | 30 | 0 | 0 | 13 | 1 | 4 | 10144.9 |
| OLMo 2 | no_stripped | cue | 1600 | 18 | 16 | 6 | 30 | 0 | 0 | 14 | 1 | 4 | 20289.8 |
| OLMo 2 | no_stripped | cue | full | 18 | 16 | 6 | 30 | 0 | 0 | 14 | 1 | 4 | 20289.8 |
| OLMo 2 | no_stripped | near_or_cue | 100 | 26 | 7 | 7 | 30 | 1 | 0 | 23 | 1 | 3 | 28568.2 |
| OLMo 2 | no_stripped | near_or_cue | 200 | 27 | 7 | 6 | 30 | 1 | 0 | 23 | 1 | 4 | 20289.8 |
| OLMo 2 | no_stripped | near_or_cue | 400 | 27 | 7 | 6 | 30 | 1 | 0 | 23 | 1 | 4 | 28568.2 |
| OLMo 2 | no_stripped | near_or_cue | 800 | 27 | 7 | 6 | 30 | 1 | 0 | 23 | 1 | 4 | 28568.2 |
| OLMo 2 | no_stripped | near_or_cue | 1600 | 27 | 7 | 6 | 30 | 1 | 0 | 23 | 1 | 4 | 40579.7 |
| OLMo 2 | no_stripped | near_or_cue | full | 27 | 7 | 6 | 30 | 1 | 0 | 23 | 1 | 4 | 40579.7 |
| Pile | all | bare | 100 | 32 | 4 | 4 | 22 | 5 | 0 | 18 | 3 | 14 | 6 |
| Pile | all | bare | 200 | 34 | 2 | 4 | 22 | 6 | 1 | 20 | 2 | 14 | 6 |
| Pile | all | bare | 400 | 34 | 2 | 4 | 22 | 6 | 1 | 20 | 2 | 14 | 6 |
| Pile | all | bare | 800 | 34 | 2 | 4 | 22 | 6 | 1 | 20 | 2 | 14 | 19.7 |
| Pile | all | bare | 1600 | 36 | 2 | 2 | 22 | 8 | 1 | 20 | 2 | 16 | 28.6 |
| Pile | all | bare | full | 36 | 2 | 2 | 22 | 8 | 1 | 20 | 2 | 16 | 28.6 |
| Pile | all | cue | 100 | 25 | 10 | 5 | 22 | 2 | 0 | 12 | 2 | 13 | 3 |
| Pile | all | cue | 200 | 25 | 10 | 5 | 22 | 2 | 0 | 12 | 2 | 13 | 3 |
| Pile | all | cue | 400 | 25 | 10 | 5 | 22 | 2 | 0 | 12 | 2 | 13 | 3 |
| Pile | all | cue | 800 | 25 | 10 | 5 | 22 | 2 | 0 | 12 | 5 | 13 | 6 |
| Pile | all | cue | 1600 | 26 | 9 | 5 | 22 | 3 | 0 | 13 | 3 | 13 | 16.7 |
| Pile | all | cue | full | 26 | 9 | 5 | 22 | 3 | 0 | 13 | 3 | 13 | 16.7 |
| Pile | all | near_or_cue | 100 | 32 | 4 | 4 | 22 | 5 | 0 | 18 | 3 | 14 | 6 |
| Pile | all | near_or_cue | 200 | 32 | 4 | 4 | 22 | 5 | 0 | 18 | 3 | 14 | 6 |
| Pile | all | near_or_cue | 400 | 32 | 4 | 4 | 22 | 5 | 0 | 18 | 3 | 14 | 6 |
| Pile | all | near_or_cue | 800 | 32 | 4 | 4 | 22 | 5 | 0 | 18 | 3 | 14 | 9 |
| Pile | all | near_or_cue | 1600 | 32 | 4 | 4 | 22 | 5 | 0 | 18 | 3 | 14 | 19.7 |
| Pile | all | near_or_cue | full | 32 | 4 | 4 | 22 | 5 | 0 | 18 | 3 | 14 | 19.7 |
| Pile | no_stripped | bare | 100 | 32 | 4 | 4 | 34 | 5 | 0 | 30 | 3 | 2 | 642.2 |
| Pile | no_stripped | bare | 200 | 34 | 2 | 4 | 34 | 6 | 2 | 32 | 3 | 2 | 642.2 |
| Pile | no_stripped | bare | 400 | 34 | 2 | 4 | 34 | 6 | 2 | 32 | 3 | 2 | 642.2 |
| Pile | no_stripped | bare | 800 | 34 | 2 | 4 | 34 | 6 | 2 | 32 | 3 | 2 | 642.2 |
| Pile | no_stripped | bare | 1600 | 36 | 2 | 2 | 34 | 6 | 2 | 32 | 3 | 4 | 642.2 |
| Pile | no_stripped | bare | full | 36 | 2 | 2 | 34 | 6 | 2 | 32 | 3 | 4 | 642.2 |
| Pile | no_stripped | cue | 100 | 25 | 10 | 5 | 34 | 3 | 0 | 24 | 2 | 1 | 421.5 |
| Pile | no_stripped | cue | 200 | 25 | 10 | 5 | 34 | 3 | 0 | 24 | 2 | 1 | 421.5 |
| Pile | no_stripped | cue | 400 | 25 | 10 | 5 | 34 | 3 | 0 | 24 | 2 | 1 | 421.5 |
| Pile | no_stripped | cue | 800 | 25 | 10 | 5 | 34 | 3 | 0 | 24 | 3 | 1 | 421.5 |
| Pile | no_stripped | cue | 1600 | 26 | 9 | 5 | 34 | 3 | 0 | 25 | 3 | 1 | 421.5 |
| Pile | no_stripped | cue | full | 26 | 9 | 5 | 34 | 3 | 0 | 25 | 3 | 1 | 421.5 |
| Pile | no_stripped | near_or_cue | 100 | 32 | 4 | 4 | 34 | 5 | 0 | 30 | 3 | 2 | 642.2 |
| Pile | no_stripped | near_or_cue | 200 | 32 | 4 | 4 | 34 | 5 | 0 | 30 | 3 | 2 | 642.2 |
| Pile | no_stripped | near_or_cue | 400 | 32 | 4 | 4 | 34 | 5 | 0 | 30 | 3 | 2 | 642.2 |
| Pile | no_stripped | near_or_cue | 800 | 32 | 4 | 4 | 34 | 6 | 0 | 30 | 3 | 2 | 642.2 |
| Pile | no_stripped | near_or_cue | 1600 | 32 | 4 | 4 | 34 | 6 | 0 | 30 | 3 | 2 | 642.2 |
| Pile | no_stripped | near_or_cue | full | 32 | 4 | 4 | 34 | 6 | 0 | 30 | 3 | 2 | 642.2 |
| RedPajama | all | bare | 100 | 32 | 1 | 7 | 16 | 7 | 2 | 15 | 4 | 17 | 5 |
| RedPajama | all | bare | 200 | 32 | 1 | 7 | 16 | 7 | 2 | 15 | 4 | 17 | 6.6 |
| RedPajama | all | bare | 400 | 32 | 1 | 7 | 16 | 9 | 2 | 15 | 4 | 17 | 44.6 |
| RedPajama | all | bare | 800 | 34 | 1 | 5 | 16 | 9 | 2 | 15 | 4 | 19 | 79.4 |
| RedPajama | all | bare | 1600 | 35 | 1 | 4 | 16 | 10 | 2 | 15 | 4 | 20 | 79.1 |
| RedPajama | all | bare | full | 35 | 1 | 4 | 16 | 10 | 2 | 15 | 4 | 20 | 79.1 |
| RedPajama | all | cue | 100 | 26 | 2 | 12 | 16 | 5 | 0 | 14 | 1 | 12 | 1 |
| RedPajama | all | cue | 200 | 26 | 2 | 12 | 16 | 5 | 0 | 14 | 1 | 12 | 1 |
| RedPajama | all | cue | 400 | 26 | 2 | 12 | 16 | 5 | 0 | 14 | 1 | 12 | 2 |
| RedPajama | all | cue | 800 | 27 | 2 | 11 | 16 | 5 | 0 | 14 | 1 | 13 | 5 |
| RedPajama | all | cue | 1600 | 29 | 2 | 9 | 16 | 5 | 0 | 14 | 1 | 15 | 6.6 |
| RedPajama | all | cue | full | 29 | 2 | 9 | 16 | 5 | 0 | 14 | 1 | 15 | 6.6 |
| RedPajama | all | near_or_cue | 100 | 32 | 1 | 7 | 16 | 7 | 2 | 15 | 4 | 17 | 5 |
| RedPajama | all | near_or_cue | 200 | 32 | 1 | 7 | 16 | 7 | 2 | 15 | 4 | 17 | 5 |
| RedPajama | all | near_or_cue | 400 | 32 | 1 | 7 | 16 | 7 | 2 | 15 | 4 | 17 | 5 |
| RedPajama | all | near_or_cue | 800 | 32 | 1 | 7 | 16 | 7 | 2 | 15 | 4 | 17 | 5 |
| RedPajama | all | near_or_cue | 1600 | 33 | 1 | 6 | 16 | 7 | 2 | 15 | 4 | 18 | 5 |
| RedPajama | all | near_or_cue | full | 33 | 1 | 6 | 16 | 7 | 2 | 15 | 4 | 18 | 5 |
| RedPajama | no_stripped | bare | 100 | 32 | 3 | 5 | 28 | 4 | 0 | 25 | 1 | 7 | 8060.8 |
| RedPajama | no_stripped | bare | 200 | 32 | 3 | 5 | 28 | 4 | 0 | 25 | 1 | 7 | 8060.8 |
| RedPajama | no_stripped | bare | 400 | 32 | 3 | 5 | 28 | 5 | 0 | 25 | 1 | 7 | 8060.8 |
| RedPajama | no_stripped | bare | 800 | 34 | 3 | 3 | 28 | 5 | 0 | 25 | 1 | 9 | 8060.8 |
| RedPajama | no_stripped | bare | 1600 | 35 | 3 | 2 | 28 | 5 | 0 | 25 | 1 | 10 | 89.3 |
| RedPajama | no_stripped | bare | full | 35 | 3 | 2 | 28 | 5 | 0 | 25 | 1 | 10 | 89.3 |
| RedPajama | no_stripped | cue | 100 | 26 | 6 | 8 | 28 | 1 | 0 | 22 | 1 | 4 | 6.6 |
| RedPajama | no_stripped | cue | 200 | 26 | 6 | 8 | 28 | 1 | 0 | 22 | 1 | 4 | 6.6 |
| RedPajama | no_stripped | cue | 400 | 26 | 6 | 8 | 28 | 1 | 0 | 22 | 1 | 4 | 6.6 |
| RedPajama | no_stripped | cue | 800 | 27 | 6 | 7 | 28 | 3 | 0 | 22 | 1 | 5 | 14891.9 |
| RedPajama | no_stripped | cue | 1600 | 29 | 6 | 5 | 28 | 3 | 0 | 22 | 1 | 7 | 22745.6 |
| RedPajama | no_stripped | cue | full | 29 | 6 | 5 | 28 | 3 | 0 | 22 | 1 | 7 | 22745.6 |
| RedPajama | no_stripped | near_or_cue | 100 | 32 | 3 | 5 | 28 | 4 | 0 | 25 | 1 | 7 | 8060.8 |
| RedPajama | no_stripped | near_or_cue | 200 | 32 | 3 | 5 | 28 | 4 | 0 | 25 | 1 | 7 | 8060.8 |
| RedPajama | no_stripped | near_or_cue | 400 | 32 | 3 | 5 | 28 | 4 | 0 | 25 | 1 | 7 | 8060.8 |
| RedPajama | no_stripped | near_or_cue | 800 | 32 | 3 | 5 | 28 | 4 | 0 | 25 | 1 | 7 | 8060.8 |
| RedPajama | no_stripped | near_or_cue | 1600 | 33 | 3 | 4 | 28 | 4 | 0 | 25 | 1 | 8 | 16121.5 |
| RedPajama | no_stripped | near_or_cue | full | 33 | 3 | 4 | 28 | 4 | 0 | 25 | 1 | 8 | 16121.5 |

## Width and matcher (strings all): median n_c (incl. 0) over docs facts, and the count

| index | matcher | 100 | 200 | 400 | 800 | 1600 | full |
|---|---|---|---|---|---|---|---|
| DCLM | bare | 0 (n=18) | 0 (n=18) | 0 (n=18) | 0 (n=18) | 0 (n=18) | 0 (n=18) |
| DCLM | cue | 0 (n=18) | 0 (n=18) | 0 (n=18) | 0 (n=18) | 0 (n=18) | 0 (n=18) |
| DCLM | near_or_cue | 0 (n=18) | 0 (n=18) | 0 (n=18) | 0 (n=18) | 0 (n=18) | 0 (n=18) |
| Dolma | bare | 3 (n=20) | 3 (n=20) | 3 (n=20) | 3 (n=20) | 3 (n=20) | 3 (n=20) |
| Dolma | cue | 1 (n=20) | 1 (n=20) | 1 (n=20) | 1 (n=20) | 1 (n=20) | 1 (n=20) |
| Dolma | near_or_cue | 3 (n=20) | 3 (n=20) | 3 (n=20) | 3 (n=20) | 3 (n=20) | 3 (n=20) |
| OLMo 2 | bare | 2 (n=18) | 2 (n=18) | 2 (n=18) | 2 (n=18) | 2 (n=18) | 2 (n=18) |
| OLMo 2 | cue | 0 (n=18) | 0 (n=18) | 0 (n=18) | 0 (n=18) | 1 (n=18) | 1 (n=18) |
| OLMo 2 | near_or_cue | 2 (n=18) | 2 (n=18) | 2 (n=18) | 2 (n=18) | 2 (n=18) | 2 (n=18) |
| Pile | bare | 5 (n=22) | 6 (n=22) | 6 (n=22) | 6 (n=22) | 8 (n=22) | 8 (n=22) |
| Pile | cue | 2 (n=22) | 2 (n=22) | 2 (n=22) | 2 (n=22) | 3 (n=22) | 3 (n=22) |
| Pile | near_or_cue | 5 (n=22) | 5 (n=22) | 5 (n=22) | 5 (n=22) | 5 (n=22) | 5 (n=22) |
| RedPajama | bare | 7 (n=16) | 7 (n=16) | 9 (n=16) | 9 (n=16) | 10 (n=16) | 10 (n=16) |
| RedPajama | cue | 5 (n=16) | 5 (n=16) | 5 (n=16) | 5 (n=16) | 5 (n=16) | 5 (n=16) |
| RedPajama | near_or_cue | 7 (n=16) | 7 (n=16) | 7 (n=16) | 7 (n=16) | 7 (n=16) | 7 (n=16) |

## Width and matcher (strings all): p10 n_c>0 over docs facts with n_c > 0, and the count (20-name preview, NOT tau)

| index | matcher | 100 | 200 | 400 | 800 | 1600 | full |
|---|---|---|---|---|---|---|---|
| DCLM | bare | 1 (n=9) | 1 (n=9) | 1 (n=9) | 1 (n=9) | 1 (n=9) | 1 (n=9) |
| DCLM | cue | 1 (n=7) | 1 (n=7) | 1 (n=7) | 1 (n=7) | 1 (n=7) | 1 (n=7) |
| DCLM | near_or_cue | 1 (n=9) | 1 (n=9) | 1 (n=9) | 1 (n=9) | 1 (n=9) | 1 (n=9) |
| Dolma | bare | 1 (n=19) | 1 (n=19) | 1 (n=19) | 1 (n=19) | 1 (n=19) | 1 (n=19) |
| Dolma | cue | 1 (n=11) | 1 (n=12) | 1 (n=12) | 1 (n=12) | 1 (n=14) | 1 (n=14) |
| Dolma | near_or_cue | 1 (n=19) | 1 (n=19) | 1 (n=19) | 1 (n=19) | 1 (n=19) | 1 (n=19) |
| OLMo 2 | bare | 1 (n=15) | 1 (n=15) | 1 (n=15) | 1 (n=15) | 1 (n=15) | 1 (n=15) |
| OLMo 2 | cue | 1 (n=9) | 1 (n=9) | 1 (n=9) | 1 (n=9) | 1 (n=10) | 1 (n=10) |
| OLMo 2 | near_or_cue | 1 (n=15) | 1 (n=15) | 1 (n=15) | 1 (n=15) | 1 (n=15) | 1 (n=15) |
| Pile | bare | 3 (n=18) | 2 (n=20) | 2 (n=20) | 2 (n=20) | 2 (n=20) | 2 (n=20) |
| Pile | cue | 2 (n=12) | 2 (n=12) | 2 (n=12) | 5 (n=12) | 3 (n=13) | 3 (n=13) |
| Pile | near_or_cue | 3 (n=18) | 3 (n=18) | 3 (n=18) | 3 (n=18) | 3 (n=18) | 3 (n=18) |
| RedPajama | bare | 4 (n=15) | 4 (n=15) | 4 (n=15) | 4 (n=15) | 4 (n=15) | 4 (n=15) |
| RedPajama | cue | 1 (n=14) | 1 (n=14) | 1 (n=14) | 1 (n=14) | 1 (n=14) | 1 (n=14) |
| RedPajama | near_or_cue | 4 (n=15) | 4 (n=15) | 4 (n=15) | 4 (n=15) | 4 (n=15) | 4 (n=15) |

## All strings vs no_stripped (full width)

| index | matcher | strings | counted | zero | unknown | median n_c (incl. 0) | p10 n_c (incl. 0) | docs facts | facts n_c>0 | p10 n_c>0 | median est_mentions |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DCLM | bare | all | 23 | 9 | 8 | 0 | 0 | 18 | 9 | 1 | 624 |
| DCLM | bare | no_stripped | 22 | 17 | 1 | 0 | 0 | 30 | 13 | 1 | 26237.8 |
| DCLM | cue | all | 14 | 11 | 15 | 0 | 0 | 18 | 7 | 1 | 10119.9 |
| DCLM | cue | no_stripped | 14 | 21 | 5 | 0 | 0 | 30 | 9 | 1 | 20239.9 |
| DCLM | near_or_cue | all | 19 | 9 | 12 | 0 | 0 | 18 | 9 | 1 | 21 |
| DCLM | near_or_cue | no_stripped | 19 | 17 | 4 | 0 | 0 | 30 | 13 | 1 | 10119.9 |
| Dolma | bare | all | 34 | 1 | 5 | 3 | 1 | 20 | 19 | 1 | 67.5 |
| Dolma | bare | no_stripped | 32 | 5 | 3 | 2 | 0 | 32 | 27 | 1 | 25184 |
| Dolma | cue | all | 25 | 6 | 9 | 1 | 0 | 20 | 14 | 1 | 34 |
| Dolma | cue | no_stripped | 24 | 13 | 3 | 1 | 0 | 32 | 19 | 1 | 12592 |
| Dolma | near_or_cue | all | 33 | 1 | 6 | 3 | 1 | 20 | 19 | 1 | 34 |
| Dolma | near_or_cue | no_stripped | 32 | 5 | 3 | 2 | 0 | 32 | 27 | 1 | 16789.4 |
| OLMo 2 | bare | all | 34 | 3 | 3 | 2 | 0 | 18 | 15 | 1 | 54.1 |
| OLMo 2 | bare | no_stripped | 31 | 7 | 2 | 1 | 0 | 30 | 23 | 1 | 10144.9 |
| OLMo 2 | cue | all | 19 | 8 | 13 | 1 | 0 | 18 | 10 | 1 | 5.2 |
| OLMo 2 | cue | no_stripped | 18 | 16 | 6 | 0 | 0 | 30 | 14 | 1 | 20289.8 |
| OLMo 2 | near_or_cue | all | 29 | 3 | 8 | 2 | 0 | 18 | 15 | 1 | 5.2 |
| OLMo 2 | near_or_cue | no_stripped | 27 | 7 | 6 | 1 | 0 | 30 | 23 | 1 | 40579.7 |
| Pile | bare | all | 36 | 2 | 2 | 8 | 1 | 22 | 20 | 2 | 28.6 |
| Pile | bare | no_stripped | 36 | 2 | 2 | 6 | 2 | 34 | 32 | 3 | 642.2 |
| Pile | cue | all | 26 | 9 | 5 | 3 | 0 | 22 | 13 | 3 | 16.7 |
| Pile | cue | no_stripped | 26 | 9 | 5 | 3 | 0 | 34 | 25 | 3 | 421.5 |
| Pile | near_or_cue | all | 32 | 4 | 4 | 5 | 0 | 22 | 18 | 3 | 19.7 |
| Pile | near_or_cue | no_stripped | 32 | 4 | 4 | 6 | 0 | 34 | 30 | 3 | 642.2 |
| RedPajama | bare | all | 35 | 1 | 4 | 10 | 2 | 16 | 15 | 4 | 79.1 |
| RedPajama | bare | no_stripped | 35 | 3 | 2 | 5 | 0 | 28 | 25 | 1 | 89.3 |
| RedPajama | cue | all | 29 | 2 | 9 | 5 | 0 | 16 | 14 | 1 | 6.6 |
| RedPajama | cue | no_stripped | 29 | 6 | 5 | 3 | 0 | 28 | 22 | 1 | 22745.6 |
| RedPajama | near_or_cue | all | 33 | 1 | 6 | 7 | 2 | 16 | 15 | 4 | 5 |
| RedPajama | near_or_cue | no_stripped | 33 | 3 | 4 | 4 | 0 | 28 | 25 | 1 | 16121.5 |

## Where matches come from (strings all, full width)

Glued share = matched_glued / mentions_matched; stripped share = matched_stripped / mentions_matched, summed over the facts of that index.

| index | matcher | mentions matched | glued | glued share | stripped | stripped share |
|---|---|---|---|---|---|---|
| DCLM | bare | 367 | 118 | 32.2% | 5 | 1.4% |
| DCLM | cue | 143 | 33 | 23.1% | 0 | 0.0% |
| DCLM | near_or_cue | 274 | 74 | 27.0% | 0 | 0.0% |
| Dolma | bare | 272 | 44 | 16.2% | 9 | 3.3% |
| Dolma | cue | 97 | 19 | 19.6% | 2 | 2.1% |
| Dolma | near_or_cue | 204 | 38 | 18.6% | 3 | 1.5% |
| OLMo 2 | bare | 416 | 130 | 31.2% | 11 | 2.6% |
| OLMo 2 | cue | 157 | 34 | 21.7% | 1 | 0.6% |
| OLMo 2 | near_or_cue | 309 | 81 | 26.2% | 2 | 0.6% |
| Pile | bare | 373 | 134 | 35.9% | 14 | 3.8% |
| Pile | cue | 195 | 76 | 39.0% | 9 | 4.6% |
| Pile | near_or_cue | 301 | 112 | 37.2% | 9 | 3.0% |
| RedPajama | bare | 294 | 110 | 37.4% | 13 | 4.4% |
| RedPajama | cue | 132 | 46 | 34.8% | 2 | 1.5% |
| RedPajama | near_or_cue | 212 | 71 | 33.5% | 2 | 0.9% |

## n_docs_dedup vs n_docs (strings all, full width, facts with a match)

| index | matcher | facts | sum n_docs | sum n_docs_dedup | facts with dedup < docs |
|---|---|---|---|---|---|
| DCLM | bare | 23 | 213 | 75 | 9 |
| DCLM | cue | 14 | 62 | 22 | 3 |
| DCLM | near_or_cue | 19 | 176 | 48 | 7 |
| Dolma | bare | 34 | 189 | 182 | 5 |
| Dolma | cue | 25 | 66 | 64 | 2 |
| Dolma | near_or_cue | 33 | 148 | 142 | 4 |
| OLMo 2 | bare | 34 | 245 | 102 | 10 |
| OLMo 2 | cue | 19 | 70 | 29 | 4 |
| OLMo 2 | near_or_cue | 29 | 196 | 66 | 8 |
| Pile | bare | 36 | 268 | 108 | 31 |
| Pile | cue | 26 | 145 | 53 | 25 |
| Pile | near_or_cue | 32 | 210 | 79 | 29 |
| RedPajama | bare | 35 | 264 | 195 | 19 |
| RedPajama | cue | 29 | 123 | 94 | 11 |
| RedPajama | near_or_cue | 33 | 185 | 140 | 16 |

Rows over the whole table where n_docs_dedup > n_docs: 0 (0 by construction: dedup counts document groups).

## URL share and truncated sides, per index

url_share = matched windows with a URL / matched windows (strings all, full width, bare). truncated_side_share = share of (window, side) pairs whose stored text on that side of the needle is shorter than 1600 characters.

| index | matched windows | with URL | url_share | truncated_side_share |
|---|---|---|---|---|
| DCLM | 367 | 367 | 100.0% | 78.9% |
| Dolma | 272 | 262 | 96.3% | 81.4% |
| OLMo 2 | 416 | 414 | 99.5% | 78.2% |
| Pile | 373 | 0 | 0.0% | 84.4% |
| RedPajama | 294 | 129 | 43.9% | 80.1% |

## Coverage (strings all)

One line per (name, index): windows fetched / windows planned, summed over names; incomplete = names with a planned window not fetched or a string never fetched.

| index | names | planned | fetched | coverage | incomplete names |
|---|---|---|---|---|---|
| DCLM | 20 | 1251 | 1251 | 100.0% | 0 |
| Dolma | 20 | 1064 | 1064 | 100.0% | 0 |
| OLMo 2 | 20 | 1312 | 1312 | 100.0% | 0 |
| Pile | 20 | 896 | 896 | 100.0% | 0 |
| RedPajama | 20 | 1093 | 1093 | 100.0% | 0 |

## Year conflicts

0 name-fact pairs have more than one year across prompts; 0 rows carry year_conflict = 1.

## Diagnostics

- extra_windows: 0
- plan_rows: 5616
- windows_rows: 5616
- plan_unfetched: 0
