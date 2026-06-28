# Automatic analysis report

## Input

- table: `results/summary/comparison_gpt2_writingprompts_table.csv`
- rows: 46

## Key findings

- The comparison table is one row per `method, n_trials` configuration.
- `minflat` has lower chosen flatness than `random` for 4/5 tested N values.
- `maxflat` is higher than `minflat` for 4/5 tested N values, which is the expected negative-control direction.

## Method summary

| method | method_family | n_trials | mean_chosen_flatness | mean_selection_gain | distinct_2 | repetition_4 | mean_runtime_seconds | mean_tokens_per_second |
|---|---|---|---|---|---|---|---|---|
| beam | baseline | - | - | - | 0.1709 | 0.7958 | 5.0732 | 34.9649 |
| centerflat_q25 | n_trial | 1.0000 | 3.0716 | 0.0000 | 0.9183 | 0.0102 | 5.8512 | 26.8272 |
| centerflat_q25 | n_trial | 4.0000 | 2.8944 | 0.1150 | 0.9149 | 0.0196 | 13.4946 | 11.7961 |
| centerflat_q25 | n_trial | 8.0000 | 2.7410 | 0.1494 | 0.9085 | 0.0353 | 23.7805 | 6.7566 |
| centerflat_q25 | n_trial | 16.0000 | 2.6234 | 0.1715 | 0.9049 | 0.0452 | 45.3515 | 3.5504 |
| centerflat_q25 | n_trial | 32.0000 | 2.5995 | 0.1975 | 0.8969 | 0.0575 | 87.7063 | 1.8371 |
| centerflat_q50 | n_trial | 1.0000 | 3.0716 | 0.0000 | 0.9183 | 0.0102 | 6.3651 | 24.7370 |
| centerflat_q50 | n_trial | 4.0000 | 3.2191 | 0.1175 | 0.8777 | 0.0248 | 14.0491 | 11.2690 |
| centerflat_q50 | n_trial | 8.0000 | 3.2879 | 0.1613 | 0.8538 | 0.0345 | 24.8498 | 6.4026 |
| centerflat_q50 | n_trial | 16.0000 | 3.2985 | 0.1959 | 0.8147 | 0.0664 | 44.6831 | 3.5754 |
| centerflat_q50 | n_trial | 32.0000 | 3.2979 | 0.2271 | 0.8081 | 0.0604 | 83.5120 | 1.8964 |
| centerflat_q75 | n_trial | 1.0000 | 3.0716 | 0.0000 | 0.9183 | 0.0102 | 5.9555 | 26.3754 |
| centerflat_q75 | n_trial | 4.0000 | 3.4485 | 0.1182 | 0.8586 | 0.0227 | 13.4451 | 11.8062 |
| centerflat_q75 | n_trial | 8.0000 | 3.5441 | 0.1606 | 0.8237 | 0.0385 | 24.2456 | 6.5353 |
| centerflat_q75 | n_trial | 16.0000 | 3.6042 | 0.1998 | 0.8125 | 0.0298 | 43.1000 | 3.7506 |
| centerflat_q75 | n_trial | 32.0000 | 3.6358 | 0.2292 | 0.7893 | 0.0367 | 71.3073 | 2.3020 |
| greedy | baseline | - | - | - | 0.1336 | 0.8326 | 3.0571 | 56.5340 |
| highprob_centerflat | n_trial | 1.0000 | 3.0716 | 0.0000 | 0.9183 | 0.0102 | 5.1310 | 31.1819 |
| highprob_centerflat | n_trial | 4.0000 | 3.4915 | 0.1111 | 0.7449 | 0.0693 | 11.1408 | 14.3796 |
| highprob_centerflat | n_trial | 8.0000 | 3.5315 | 0.1470 | 0.6609 | 0.1484 | 19.7704 | 8.2502 |
| highprob_centerflat | n_trial | 16.0000 | 3.5356 | 0.1786 | 0.6233 | 0.1881 | 36.6442 | 4.4540 |
| highprob_centerflat | n_trial | 32.0000 | 3.5191 | 0.2098 | 0.5838 | 0.2208 | 70.5201 | 2.3235 |
| maxflat | n_trial | 1.0000 | 3.0716 | 0.0000 | 0.9183 | 0.0102 | 6.4511 | 24.5191 |
| maxflat | n_trial | 4.0000 | 3.8119 | 0.1158 | 0.8601 | 0.0228 | 13.3891 | 11.0860 |
| maxflat | n_trial | 8.0000 | 3.9797 | 0.1536 | 0.8569 | 0.0126 | 22.6823 | 6.4465 |
| maxflat | n_trial | 16.0000 | 4.0827 | 0.1778 | 0.8386 | 0.0137 | 44.8167 | 3.2838 |
| maxflat | n_trial | 32.0000 | 4.1413 | 0.1915 | 0.8495 | 0.0174 | 87.7609 | 1.6676 |
| midflat | n_trial | 1.0000 | 3.0716 | 0.0000 | 0.9183 | 0.0102 | 7.1752 | 22.6055 |
| midflat | n_trial | 4.0000 | 3.2191 | 0.1175 | 0.8777 | 0.0248 | 16.3716 | 9.8181 |
| midflat | n_trial | 8.0000 | 3.2879 | 0.1613 | 0.8538 | 0.0345 | 26.5124 | 6.0363 |
| midflat | n_trial | 16.0000 | 3.2985 | 0.1959 | 0.8147 | 0.0664 | 46.9590 | 3.4277 |
| midflat | n_trial | 32.0000 | 3.2979 | 0.2271 | 0.8081 | 0.0604 | 84.0788 | 1.8852 |
| minflat | n_trial | 1.0000 | 3.0716 | 0.0000 | 0.9183 | 0.0102 | 12.6592 | 14.9868 |
| minflat | n_trial | 4.0000 | 1.8946 | 0.0801 | 0.9462 | 0.0181 | 24.9967 | 7.1272 |
| minflat | n_trial | 8.0000 | 1.7489 | 0.0897 | 0.8807 | 0.0450 | 44.8876 | 4.0574 |
| minflat | n_trial | 16.0000 | 1.6851 | 0.1005 | 0.8063 | 0.0924 | 84.4651 | 2.1631 |
| minflat | n_trial | 32.0000 | 1.6558 | 0.1063 | 0.8146 | 0.0779 | 116.5778 | 1.4860 |
| random | n_trial | 1.0000 | 3.0716 | 0.0000 | 0.9183 | 0.0102 | 12.7492 | 14.7071 |
| random | n_trial | 4.0000 | 3.0611 | 0.1189 | 0.9235 | 0.0096 | 24.2226 | 7.1436 |
| random | n_trial | 8.0000 | 3.0617 | 0.1615 | 0.9210 | 0.0073 | 42.0467 | 4.1986 |
| random | n_trial | 16.0000 | 3.0857 | 0.1980 | 0.9166 | 0.0070 | 74.4968 | 2.2835 |
| random | n_trial | 32.0000 | 3.0553 | 0.2256 | 0.9241 | 0.0090 | 142.7891 | 1.1746 |
| sampling | baseline | - | - | - | 0.9229 | 0.0075 | 4.0064 | 49.6683 |
| temperature | baseline | - | - | - | 0.9911 | 0.0003 | 4.4351 | 48.4516 |
| top_k | baseline | - | - | - | 0.8385 | 0.0181 | 4.4488 | 46.4334 |
| top_p | baseline | - | - | - | 0.9750 | 0.0010 | 4.5629 | 45.8897 |

## Flatness-N vs Random-N

| n_trials | random | minflat | minflat_minus_random | relative_delta |
|---|---|---|---|---|
| 1.0000 | 3.0716 | 3.0716 | 0.0000 | 0.0000 |
| 4.0000 | 3.0611 | 1.8946 | -1.1666 | -0.3811 |
| 8.0000 | 3.0617 | 1.7489 | -1.3128 | -0.4288 |
| 16.0000 | 3.0857 | 1.6851 | -1.4006 | -0.4539 |
| 32.0000 | 3.0553 | 1.6558 | -1.3995 | -0.4581 |

## Flatness control table

| n_trials | random | minflat | maxflat | minflat_minus_random | maxflat_minus_random | maxflat_minus_minflat |
|---|---|---|---|---|---|---|
| 1.0000 | 3.0716 | 3.0716 | 3.0716 | 0.0000 | 0.0000 | 0.0000 |
| 4.0000 | 3.0611 | 1.8946 | 3.8119 | -1.1666 | 0.7508 | 1.9173 |
| 8.0000 | 3.0617 | 1.7489 | 3.9797 | -1.3128 | 0.9180 | 2.2308 |
| 16.0000 | 3.0857 | 1.6851 | 4.0827 | -1.4006 | 0.9970 | 2.3976 |
| 32.0000 | 3.0553 | 1.6558 | 4.1413 | -1.3995 | 1.0860 | 2.4855 |

## N dependence

| method | n_trials | mean_chosen_flatness | mean_candidate_flatness | mean_selection_gain | mean_candidate_unique_count | distinct_2 | repetition_4 | mean_runtime_seconds |
|---|---|---|---|---|---|---|---|---|
| maxflat | 1.0000 | 3.0716 | 3.0716 | 0.0000 | 1.0000 | 0.9183 | 0.0102 | 6.4511 |
| maxflat | 4.0000 | 3.8119 | 3.4220 | 0.1158 | 3.2057 | 0.8601 | 0.0228 | 13.3891 |
| maxflat | 8.0000 | 3.9797 | 3.4543 | 0.1536 | 5.5524 | 0.8569 | 0.0126 | 22.6823 |
| maxflat | 16.0000 | 4.0827 | 3.4807 | 0.1778 | 9.3301 | 0.8386 | 0.0137 | 44.8167 |
| maxflat | 32.0000 | 4.1413 | 3.4898 | 0.1915 | 15.1745 | 0.8495 | 0.0174 | 87.7609 |
| minflat | 1.0000 | 3.0716 | 3.0716 | 0.0000 | 1.0000 | 0.9183 | 0.0102 | 12.6592 |
| minflat | 4.0000 | 1.8946 | 2.0720 | 0.0801 | 3.7024 | 0.9462 | 0.0181 | 24.9967 |
| minflat | 8.0000 | 1.7489 | 1.9312 | 0.0897 | 6.8917 | 0.8807 | 0.0450 | 44.8876 |
| minflat | 16.0000 | 1.6851 | 1.8821 | 0.1005 | 12.7602 | 0.8063 | 0.0924 | 84.4651 |
| minflat | 32.0000 | 1.6558 | 1.8605 | 0.1063 | 23.5158 | 0.8146 | 0.0779 | 116.5778 |
| random | 1.0000 | 3.0716 | 3.0716 | 0.0000 | 1.0000 | 0.9183 | 0.0102 | 12.7492 |
| random | 4.0000 | 3.0611 | 3.0610 | 0.1189 | 3.3535 | 0.9235 | 0.0096 | 24.2226 |
| random | 8.0000 | 3.0617 | 3.0651 | 0.1615 | 6.0553 | 0.9210 | 0.0073 | 42.0467 |
| random | 16.0000 | 3.0857 | 3.0823 | 0.1980 | 10.7131 | 0.9166 | 0.0070 | 74.4968 |
| random | 32.0000 | 3.0553 | 3.0555 | 0.2256 | 19.3640 | 0.9241 | 0.0090 | 142.7891 |

## Correlation with chosen flatness

| metric | corr_with_mean_chosen_flatness |
|---|---|
| mean_selection_gain | 0.2953 |
| distinct_1 | -0.7481 |
| distinct_2 | -0.3054 |
| distinct_3 | -0.0341 |
| repetition_4 | 0.0051 |
| mean_runtime_seconds | -0.1983 |
| mean_tokens_per_second | 0.0230 |

## Interpretation checklist

- Does `minflat` reduce chosen flatness relative to `random` at the same N?
- Does `maxflat` move in the opposite direction as a negative control?
- Does increasing N change chosen flatness monotonically or approximately?
- Does lower flatness trade off with distinct-n or repetition?
- How quickly does runtime increase as N grows?
