# Automatic analysis report

## Input

- table: `results/summary/comparison_smoke_table.csv`
- rows: 15

## Key findings

- The comparison table is one row per `method, n_trials` configuration.
- `minflat` has lower chosen flatness than `random` for 3/3 tested N values.
- `maxflat` is higher than `minflat` for 2/3 tested N values, which is the expected negative-control direction.

## Method summary

| method | method_family | n_trials | mean_chosen_flatness | mean_selection_gain | distinct_2 | repetition_4 | mean_runtime_seconds | mean_tokens_per_second |
|---|---|---|---|---|---|---|---|---|
| beam | baseline | - | - | - | 0.6869 | 0.2787 | 1.8634 | 24.5418 |
| greedy | baseline | - | - | - | 0.6923 | 0.2879 | 1.5142 | 41.3021 |
| maxflat | n_trial | 1.0000 | 2.0563 | 0.0000 | 0.9865 | 0.0018 | 2.2039 | 33.7786 |
| maxflat | n_trial | 4.0000 | 2.2468 | 0.0725 | 0.9795 | 0.0019 | 3.4638 | 17.6393 |
| maxflat | n_trial | 8.0000 | 2.4746 | 0.1026 | 0.9850 | 0.0000 | 6.5389 | 10.0193 |
| minflat | n_trial | 1.0000 | 2.0563 | 0.0000 | 0.9865 | 0.0018 | 3.0714 | 27.8170 |
| minflat | n_trial | 4.0000 | 1.5346 | 0.0656 | 0.9951 | 0.0000 | 4.0410 | 17.0446 |
| minflat | n_trial | 8.0000 | 1.2218 | 0.0675 | 0.9263 | 0.0563 | 6.5944 | 9.9866 |
| random | n_trial | 1.0000 | 2.0574 | 0.0000 | 0.9865 | 0.0018 | 3.9371 | 17.6216 |
| random | n_trial | 4.0000 | 1.9724 | 0.0710 | 0.9958 | 0.0000 | 3.8118 | 17.3043 |
| random | n_trial | 8.0000 | 2.0102 | 0.0998 | 0.9943 | 0.0000 | 9.3204 | 8.6861 |
| sampling | baseline | - | - | - | 0.9688 | 0.0262 | 1.7970 | 39.1516 |
| temperature | baseline | - | - | - | 1.0000 | 0.0000 | 1.6592 | 37.3430 |
| top_k | baseline | - | - | - | 0.9926 | 0.0000 | 1.7987 | 37.7574 |
| top_p | baseline | - | - | - | 0.9962 | 0.0000 | 1.7065 | 39.0717 |

## Flatness-N vs Random-N

| n_trials | random | minflat | minflat_minus_random | relative_delta |
|---|---|---|---|---|
| 1.0000 | 2.0574 | 2.0563 | -0.0011 | -0.0005 |
| 4.0000 | 1.9724 | 1.5346 | -0.4378 | -0.2220 |
| 8.0000 | 2.0102 | 1.2218 | -0.7884 | -0.3922 |

## Flatness control table

| n_trials | random | minflat | maxflat | minflat_minus_random | maxflat_minus_random | maxflat_minus_minflat |
|---|---|---|---|---|---|---|
| 1.0000 | 2.0574 | 2.0563 | 2.0563 | -0.0011 | -0.0011 | 0.0000 |
| 4.0000 | 1.9724 | 1.5346 | 2.2468 | -0.4378 | 0.2745 | 0.7122 |
| 8.0000 | 2.0102 | 1.2218 | 2.4746 | -0.7884 | 0.4643 | 1.2527 |

## N dependence

| method | n_trials | mean_chosen_flatness | mean_candidate_flatness | mean_selection_gain | mean_candidate_unique_count | distinct_2 | repetition_4 | mean_runtime_seconds |
|---|---|---|---|---|---|---|---|---|
| maxflat | 1.0000 | 2.0563 | 2.0563 | 0.0000 | 1.0000 | 0.9865 | 0.0018 | 2.2039 |
| maxflat | 4.0000 | 2.2468 | 2.0203 | 0.0725 | 3.6036 | 0.9795 | 0.0019 | 3.4638 |
| maxflat | 8.0000 | 2.4746 | 2.1469 | 0.1026 | 6.7990 | 0.9850 | 0.0000 | 6.5389 |
| minflat | 1.0000 | 2.0563 | 2.0563 | 0.0000 | 1.0000 | 0.9865 | 0.0018 | 3.0714 |
| minflat | 4.0000 | 1.5346 | 1.7314 | 0.0656 | 3.5803 | 0.9951 | 0.0000 | 4.0410 |
| minflat | 8.0000 | 1.2218 | 1.3910 | 0.0675 | 6.8147 | 0.9263 | 0.0563 | 6.5944 |
| random | 1.0000 | 2.0574 | 2.0574 | 0.0000 | 1.0000 | 0.9865 | 0.0018 | 3.9371 |
| random | 4.0000 | 1.9724 | 1.9846 | 0.0710 | 3.5292 | 0.9958 | 0.0000 | 3.8118 |
| random | 8.0000 | 2.0102 | 2.0062 | 0.0998 | 6.6799 | 0.9943 | 0.0000 | 9.3204 |

## Correlation with chosen flatness

| metric | corr_with_mean_chosen_flatness |
|---|---|
| mean_selection_gain | 0.0180 |
| distinct_1 | -0.5476 |
| distinct_2 | 0.6022 |
| distinct_3 | 0.6808 |
| repetition_4 | -0.7359 |
| mean_runtime_seconds | -0.1161 |
| mean_tokens_per_second | 0.1631 |

## Interpretation checklist

- Does `minflat` reduce chosen flatness relative to `random` at the same N?
- Does `maxflat` move in the opposite direction as a negative control?
- Does increasing N change chosen flatness monotonically or approximately?
- Does lower flatness trade off with distinct-n or repetition?
- How quickly does runtime increase as N grows?
