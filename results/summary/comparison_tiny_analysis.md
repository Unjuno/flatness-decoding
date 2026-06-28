# Automatic analysis report

## Input

- table: `results/summary/comparison_tiny_table.csv`
- rows: 10

## Key findings

- The comparison table is one row per `method, n_trials` configuration.
- `minflat` has lower chosen flatness than `random` for 1/2 tested N values.
- `maxflat` is higher than `minflat` for 1/2 tested N values, which is the expected negative-control direction.

## Method summary

| method | method_family | n_trials | mean_chosen_flatness | mean_selection_gain | distinct_2 | repetition_4 | mean_runtime_seconds | mean_tokens_per_second |
|---|---|---|---|---|---|---|---|---|
| beam | baseline | - | - | - | 0.9167 | 0.0000 | 0.9804 | 28.4201 |
| greedy | baseline | - | - | - | 0.9167 | 0.0000 | 0.4960 | 41.7312 |
| maxflat | n_trial | 1.0000 | 3.0113 | 0.0000 | 0.8788 | 0.0000 | 0.6244 | 32.0325 |
| maxflat | n_trial | 2.0000 | 3.2198 | 0.0468 | 0.7568 | 0.1515 | 0.6644 | 30.1415 |
| minflat | n_trial | 1.0000 | 3.0113 | 0.0000 | 0.8788 | 0.0000 | 0.6721 | 29.7702 |
| minflat | n_trial | 2.0000 | 2.9781 | 0.0615 | 1.0000 | 0.0000 | 0.6603 | 30.2937 |
| random | n_trial | 1.0000 | 3.0113 | 0.0000 | 0.8788 | 0.0000 | 1.4290 | 15.4308 |
| random | n_trial | 2.0000 | 3.1093 | 0.0270 | 0.9211 | 0.0000 | 1.3396 | 16.7071 |
| sampling | baseline | - | - | - | 1.0000 | 0.0000 | 0.3701 | 54.3639 |
| top_p | baseline | - | - | - | 0.9667 | 0.0000 | 0.3156 | 63.3832 |

## Flatness-N vs Random-N

| n_trials | random | minflat | minflat_minus_random | relative_delta |
|---|---|---|---|---|
| 1.0000 | 3.0113 | 3.0113 | 0.0000 | 0.0000 |
| 2.0000 | 3.1093 | 2.9781 | -0.1312 | -0.0422 |

## Flatness control table

| n_trials | random | minflat | maxflat | minflat_minus_random | maxflat_minus_random | maxflat_minus_minflat |
|---|---|---|---|---|---|---|
| 1.0000 | 3.0113 | 3.0113 | 3.0113 | 0.0000 | 0.0000 | 0.0000 |
| 2.0000 | 3.1093 | 2.9781 | 3.2198 | -0.1312 | 0.1105 | 0.2417 |

## N dependence

| method | n_trials | mean_chosen_flatness | mean_candidate_flatness | mean_selection_gain | mean_candidate_unique_count | distinct_2 | repetition_4 | mean_runtime_seconds |
|---|---|---|---|---|---|---|---|---|
| maxflat | 1.0000 | 3.0113 | 3.0113 | 0.0000 | 1.0000 | 0.8788 | 0.0000 | 0.6244 |
| maxflat | 2.0000 | 3.2198 | 3.0646 | 0.0468 | 1.5000 | 0.7568 | 0.1515 | 0.6644 |
| minflat | 1.0000 | 3.0113 | 3.0113 | 0.0000 | 1.0000 | 0.8788 | 0.0000 | 0.6721 |
| minflat | 2.0000 | 2.9781 | 3.1685 | 0.0615 | 1.8250 | 1.0000 | 0.0000 | 0.6603 |
| random | 1.0000 | 3.0113 | 3.0113 | 0.0000 | 1.0000 | 0.8788 | 0.0000 | 1.4290 |
| random | 2.0000 | 3.1093 | 3.1070 | 0.0270 | 1.5000 | 0.9211 | 0.0000 | 1.3396 |

## Correlation with chosen flatness

| metric | corr_with_mean_chosen_flatness |
|---|---|
| mean_selection_gain | 0.3393 |
| distinct_1 | -0.5869 |
| distinct_2 | -0.7557 |
| distinct_3 | -0.8028 |
| repetition_4 | 0.8739 |
| mean_runtime_seconds | 0.0138 |
| mean_tokens_per_second | -0.0336 |

## Interpretation checklist

- Does `minflat` reduce chosen flatness relative to `random` at the same N?
- Does `maxflat` move in the opposite direction as a negative control?
- Does increasing N change chosen flatness monotonically or approximately?
- Does lower flatness trade off with distinct-n or repetition?
- How quickly does runtime increase as N grows?
