# Automatic analysis report

## Input

- table: `results/summary/comparison_distilgpt2_main_table.csv`
- rows: 21

## Key findings

- The comparison table is one row per `method, n_trials` configuration.

## Method summary

| method | method_family | n_trials | mean_chosen_flatness | mean_selection_gain | distinct_2 | repetition_4 | mean_runtime_seconds | mean_tokens_per_second |
|---|---|---|---|---|---|---|---|---|
| beam | baseline | - | - | - | 0.2064 | 0.7781 | 2.5663 | 47.9155 |
| greedy | baseline | - | - | - | 0.2006 | 0.7749 | 1.7647 | 83.3718 |
| maxflat | n_trial | 1.0000 | - | - | 0.9233 | 0.0193 | 7.0603 | 20.3853 |
| maxflat | n_trial | 4.0000 | - | - | 0.9232 | 0.0122 | 10.4940 | 10.7178 |
| maxflat | n_trial | 8.0000 | - | - | 0.9048 | 0.0300 | 14.2283 | 6.7126 |
| maxflat | n_trial | 16.0000 | - | - | 0.8715 | 0.0360 | 19.5949 | 3.8889 |
| maxflat | n_trial | 32.0000 | - | - | 0.7957 | 0.0814 | 33.5067 | 2.1025 |
| minflat | n_trial | 1.0000 | - | - | 0.9233 | 0.0193 | 5.7978 | 25.5630 |
| minflat | n_trial | 4.0000 | - | - | 0.9079 | 0.0559 | 11.4343 | 12.5978 |
| minflat | n_trial | 8.0000 | - | - | 0.8729 | 0.0711 | 20.3881 | 6.6769 |
| minflat | n_trial | 16.0000 | - | - | 0.8412 | 0.0864 | 35.8344 | 3.7254 |
| minflat | n_trial | 32.0000 | - | - | 0.8538 | 0.0809 | 65.5751 | 1.9779 |
| random | n_trial | 1.0000 | - | - | 0.9233 | 0.0193 | 6.5708 | 23.5667 |
| random | n_trial | 4.0000 | - | - | 0.9379 | 0.0122 | 11.2388 | 11.9381 |
| random | n_trial | 8.0000 | - | - | 0.9249 | 0.0242 | 18.0572 | 7.2886 |
| random | n_trial | 16.0000 | - | - | 0.9291 | 0.0181 | 31.6734 | 3.9737 |
| random | n_trial | 32.0000 | - | - | 0.9270 | 0.0219 | 57.3199 | 2.1734 |
| sampling | baseline | - | - | - | 0.9351 | 0.0141 | 2.6819 | 60.2555 |
| temperature | baseline | - | - | - | 0.9873 | 0.0003 | 2.6606 | 58.6183 |
| top_k | baseline | - | - | - | 0.9069 | 0.0131 | 2.7926 | 57.1344 |
| top_p | baseline | - | - | - | 0.9715 | 0.0055 | 2.9024 | 54.9668 |

## Flatness-N vs Random-N

_No data._

## Flatness control table

_No data._

## N dependence

| method | n_trials | mean_chosen_flatness | mean_candidate_flatness | mean_selection_gain | mean_candidate_unique_count | distinct_2 | repetition_4 | mean_runtime_seconds |
|---|---|---|---|---|---|---|---|---|
| maxflat | 1.0000 | - | - | - | - | 0.9233 | 0.0193 | 7.0603 |
| maxflat | 4.0000 | - | - | - | - | 0.9232 | 0.0122 | 10.4940 |
| maxflat | 8.0000 | - | - | - | - | 0.9048 | 0.0300 | 14.2283 |
| maxflat | 16.0000 | - | - | - | - | 0.8715 | 0.0360 | 19.5949 |
| maxflat | 32.0000 | - | - | - | - | 0.7957 | 0.0814 | 33.5067 |
| minflat | 1.0000 | - | - | - | - | 0.9233 | 0.0193 | 5.7978 |
| minflat | 4.0000 | - | - | - | - | 0.9079 | 0.0559 | 11.4343 |
| minflat | 8.0000 | - | - | - | - | 0.8729 | 0.0711 | 20.3881 |
| minflat | 16.0000 | - | - | - | - | 0.8412 | 0.0864 | 35.8344 |
| minflat | 32.0000 | - | - | - | - | 0.8538 | 0.0809 | 65.5751 |
| random | 1.0000 | - | - | - | - | 0.9233 | 0.0193 | 6.5708 |
| random | 4.0000 | - | - | - | - | 0.9379 | 0.0122 | 11.2388 |
| random | 8.0000 | - | - | - | - | 0.9249 | 0.0242 | 18.0572 |
| random | 16.0000 | - | - | - | - | 0.9291 | 0.0181 | 31.6734 |
| random | 32.0000 | - | - | - | - | 0.9270 | 0.0219 | 57.3199 |

## Correlation with chosen flatness

| metric | corr_with_mean_chosen_flatness |
|---|---|
| mean_selection_gain | - |
| distinct_1 | - |
| distinct_2 | - |
| distinct_3 | - |
| repetition_4 | - |
| mean_runtime_seconds | - |
| mean_tokens_per_second | - |

## Interpretation checklist

- Does `minflat` reduce chosen flatness relative to `random` at the same N?
- Does `maxflat` move in the opposite direction as a negative control?
- Does increasing N change chosen flatness monotonically or approximately?
- Does lower flatness trade off with distinct-n or repetition?
- How quickly does runtime increase as N grows?
