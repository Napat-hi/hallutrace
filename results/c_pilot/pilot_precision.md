# Pilot precision (hand-labelled packet)

Numbers only, no snippets. precision = correct / (labelled - unclear). `bare` uses every labelled row, `cue` only rows with matched_cue = 1. 'scored' = labelled minus unclear; a figure with fewer than 5 scored rows is shown as 'too few rows'. Labelled rows: 60. The packet over-samples far bands, so only the weighted rule figure estimates a rule's real precision.

## Per band and matcher

| band | bare | cue |
|---|---|---|
| 0-100 | 0.850 (correct 17 of 20 scored; labelled 20, unclear 0) | 1.000 (correct 5 of 5 scored; labelled 5, unclear 0) |
| 101-200 | 0.600 (correct 6 of 10 scored; labelled 10, unclear 0) | 0.800 (correct 4 of 5 scored; labelled 5, unclear 0) |
| 201-400 | 0.667 (correct 6 of 9 scored; labelled 10, unclear 1) | too few rows (correct 4 of 4 scored; labelled 5, unclear 1) |
| 401-800 | 0.700 (correct 7 of 10 scored; labelled 10, unclear 0) | 1.000 (correct 5 of 5 scored; labelled 5, unclear 0) |
| 801-1600 | 0.500 (correct 5 of 10 scored; labelled 10, unclear 0) | 1.000 (correct 5 of 5 scored; labelled 5, unclear 0) |
| >1600 | too few rows (correct 0 of 0 scored; labelled 0, unclear 0) | too few rows (correct 0 of 0 scored; labelled 0, unclear 0) |

## Per fact

| fact | bare | cue |
|---|---|---|
| birth | 0.850 (correct 17 of 20 scored; labelled 20, unclear 0) | 1.000 (correct 9 of 9 scored; labelled 9, unclear 0) |
| death | 0.615 (correct 24 of 39 scored; labelled 40, unclear 1) | 0.933 (correct 14 of 15 scored; labelled 16, unclear 1) |

## Per query source

| query source | bare | cue |
|---|---|---|
| alias | 1.000 (correct 10 of 10 scored; labelled 10, unclear 0) | 1.000 (correct 6 of 6 scored; labelled 6, unclear 0) |
| folded | too few rows (correct 1 of 1 scored; labelled 1, unclear 0) | too few rows (correct 1 of 1 scored; labelled 1, unclear 0) |
| name | 0.744 (correct 29 of 39 scored; labelled 39, unclear 0) | 1.000 (correct 15 of 15 scored; labelled 15, unclear 0) |
| stripped | 0.111 (correct 1 of 9 scored; labelled 10, unclear 1) | too few rows (correct 1 of 2 scored; labelled 3, unclear 1) |

## Per candidate rule (width x matcher)

A band is inside width W if its upper bound is at most W (full: all bands). Unweighted = correct / scored over all labelled rows inside W. Weighted = sum over inside bands of (population share of the band among inside candidates for that matcher) x (precision within the band). A band with no scored rows is excluded and the weights of the remaining bands are renormalised to sum to 1; excluded bands are listed.

| width | matcher | labelled rows | scored rows | unweighted | weighted | bands used | bands excluded |
|---|---|---|---|---|---|---|---|
| 100 | bare | 20 | 20 | 0.850 (correct 17 of 20 scored; labelled 20, unclear 0) | 0.850 | 0-100 | none |
| 100 | cue | 5 | 5 | 1.000 (correct 5 of 5 scored; labelled 5, unclear 0) | 1.000 | 0-100 | none |
| 200 | bare | 30 | 30 | 0.767 (correct 23 of 30 scored; labelled 30, unclear 0) | 0.823 | 0-100, 101-200 | none |
| 200 | cue | 10 | 10 | 0.900 (correct 9 of 10 scored; labelled 10, unclear 0) | 0.994 | 0-100, 101-200 | none |
| 400 | bare | 40 | 39 | 0.744 (correct 29 of 39 scored; labelled 40, unclear 1) | 0.796 | 0-100, 101-200, 201-400 | none |
| 400 | cue | 15 | 14 | 0.929 (correct 13 of 14 scored; labelled 15, unclear 1) | 0.996 | 0-100, 101-200, 201-400 | none |
| 800 | bare | 50 | 49 | 0.735 (correct 36 of 49 scored; labelled 50, unclear 1) | 0.786 | 0-100, 101-200, 201-400, 401-800 | none |
| 800 | cue | 20 | 19 | 0.947 (correct 18 of 19 scored; labelled 20, unclear 1) | 0.996 | 0-100, 101-200, 201-400, 401-800 | none |
| 1600 | bare | 60 | 59 | 0.695 (correct 41 of 59 scored; labelled 60, unclear 1) | 0.745 | 0-100, 101-200, 201-400, 401-800, 801-1600 | none |
| 1600 | cue | 25 | 24 | 0.958 (correct 23 of 24 scored; labelled 25, unclear 1) | 0.997 | 0-100, 101-200, 201-400, 401-800, 801-1600 | none |
| full | bare | 60 | 59 | 0.695 (correct 41 of 59 scored; labelled 60, unclear 1) | 0.745 | 0-100, 101-200, 201-400, 401-800, 801-1600 | >1600 |
| full | cue | 25 | 24 | 0.958 (correct 23 of 24 scored; labelled 25, unclear 1) | 0.997 | 0-100, 101-200, 201-400, 401-800, 801-1600 | >1600 |
