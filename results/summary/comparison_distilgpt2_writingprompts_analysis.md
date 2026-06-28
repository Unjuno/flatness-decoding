# Automatic analysis report

## Input

- table: `results/summary/comparison_distilgpt2_writingprompts_table.csv`
- rows: 46

## Key findings

- The comparison table is one row per `method, n_trials` configuration.
- `minflat` has lower chosen flatness than `random` for 4/5 tested N values.
- `maxflat` is higher than `minflat` for 4/5 tested N values, which is the expected negative-control direction.

## Method summary

| method | method_family | n_trials | mean_chosen_flatness | mean_selection_gain | distinct_2 | repetition_4 | mean_runtime_seconds | mean_tokens_per_second |
|---|---|---|---|---|---|---|---|---|
| beam | baseline | - | - | - | 0.1212 | 0.8637 | 2.8890 | 41.0254 |
| centerflat_q25 | n_trial | 1.0000 | 3.1478 | 0.0000 | 0.9074 | 0.0223 | 7.2027 | 19.7622 |
| centerflat_q25 | n_trial | 4.0000 | 3.0749 | 0.0974 | 0.7884 | 0.1327 | 13.1028 | 10.2686 |
| centerflat_q25 | n_trial | 8.0000 | 2.9450 | 0.1220 | 0.7322 | 0.2059 | 20.8169 | 6.2812 |
| centerflat_q25 | n_trial | 16.0000 | 2.8702 | 0.1468 | 0.7001 | 0.2351 | 35.4314 | 3.6094 |
| centerflat_q25 | n_trial | 32.0000 | 2.8537 | 0.1674 | 0.6759 | 0.2670 | 64.7034 | 1.9523 |
| centerflat_q50 | n_trial | 1.0000 | 3.1478 | 0.0000 | 0.9074 | 0.0223 | 7.2318 | 19.6796 |
| centerflat_q50 | n_trial | 4.0000 | 3.2557 | 0.1004 | 0.8024 | 0.1073 | 13.2453 | 10.2495 |
| centerflat_q50 | n_trial | 8.0000 | 3.3062 | 0.1365 | 0.7417 | 0.1521 | 20.7209 | 6.2883 |
| centerflat_q50 | n_trial | 16.0000 | 3.3145 | 0.1619 | 0.6927 | 0.1929 | 35.0126 | 3.6211 |
| centerflat_q50 | n_trial | 32.0000 | 3.3150 | 0.1768 | 0.6373 | 0.2624 | 64.2743 | 1.9504 |
| centerflat_q75 | n_trial | 1.0000 | 3.1478 | 0.0000 | 0.9074 | 0.0223 | 7.2256 | 19.5326 |
| centerflat_q75 | n_trial | 4.0000 | 3.4041 | 0.1030 | 0.8246 | 0.0649 | 12.9000 | 10.2538 |
| centerflat_q75 | n_trial | 8.0000 | 3.4668 | 0.1420 | 0.7837 | 0.0791 | 20.7975 | 6.2719 |
| centerflat_q75 | n_trial | 16.0000 | 3.4942 | 0.1734 | 0.7462 | 0.1079 | 34.8982 | 3.6336 |
| centerflat_q75 | n_trial | 32.0000 | 3.4855 | 0.1991 | 0.7148 | 0.1337 | 64.9674 | 1.9483 |
| greedy | baseline | - | - | - | 0.1282 | 0.8507 | 2.0891 | 65.1841 |
| highprob_centerflat | n_trial | 1.0000 | 3.1478 | 0.0000 | 0.9074 | 0.0223 | 7.2146 | 19.7880 |
| highprob_centerflat | n_trial | 4.0000 | 3.3393 | 0.0822 | 0.6269 | 0.2401 | 13.0931 | 10.2553 |
| highprob_centerflat | n_trial | 8.0000 | 3.3584 | 0.0998 | 0.4993 | 0.3889 | 20.7818 | 6.3007 |
| highprob_centerflat | n_trial | 16.0000 | 3.3590 | 0.1242 | 0.4675 | 0.4148 | 34.5368 | 3.6582 |
| highprob_centerflat | n_trial | 32.0000 | 3.3461 | 0.1411 | 0.4183 | 0.4754 | 62.2775 | 2.0065 |
| maxflat | n_trial | 1.0000 | 3.1478 | 0.0000 | 0.9074 | 0.0223 | 9.8593 | 17.9163 |
| maxflat | n_trial | 4.0000 | 3.6848 | 0.1043 | 0.8783 | 0.0159 | 15.7482 | 9.1719 |
| maxflat | n_trial | 8.0000 | 3.8013 | 0.1390 | 0.8662 | 0.0113 | 22.1860 | 5.8204 |
| maxflat | n_trial | 16.0000 | 3.8901 | 0.1668 | 0.8311 | 0.0216 | 32.3772 | 3.3454 |
| maxflat | n_trial | 32.0000 | 3.9499 | 0.1904 | 0.7883 | 0.0564 | 49.2001 | 1.8566 |
| midflat | n_trial | 1.0000 | 3.1478 | 0.0000 | 0.9074 | 0.0223 | 9.6611 | 18.1726 |
| midflat | n_trial | 4.0000 | 3.2557 | 0.1004 | 0.8024 | 0.1073 | 18.1570 | 9.0604 |
| midflat | n_trial | 8.0000 | 3.3062 | 0.1365 | 0.7417 | 0.1521 | 26.8910 | 5.6411 |
| midflat | n_trial | 16.0000 | 3.3145 | 0.1619 | 0.6927 | 0.1929 | 40.6161 | 3.4483 |
| midflat | n_trial | 32.0000 | 3.3150 | 0.1768 | 0.6373 | 0.2624 | 71.7657 | 1.8419 |
| minflat | n_trial | 1.0000 | 3.1478 | 0.0000 | 0.9074 | 0.0223 | 5.3896 | 23.4304 |
| minflat | n_trial | 4.0000 | 2.1449 | 0.0807 | 0.8612 | 0.0900 | 11.4337 | 11.3307 |
| minflat | n_trial | 8.0000 | 1.9155 | 0.0936 | 0.8127 | 0.1079 | 18.7473 | 6.9028 |
| minflat | n_trial | 16.0000 | 1.8171 | 0.1028 | 0.7953 | 0.1116 | 38.2321 | 3.5871 |
| minflat | n_trial | 32.0000 | 1.7507 | 0.1154 | 0.8455 | 0.0670 | 89.5645 | 1.6109 |
| random | n_trial | 1.0000 | 3.1478 | 0.0000 | 0.9074 | 0.0223 | 9.2535 | 16.4548 |
| random | n_trial | 4.0000 | 3.1406 | 0.1101 | 0.9021 | 0.0279 | 17.3477 | 8.4058 |
| random | n_trial | 8.0000 | 3.1541 | 0.1516 | 0.8877 | 0.0373 | 26.8845 | 5.2112 |
| random | n_trial | 16.0000 | 3.1515 | 0.1875 | 0.8959 | 0.0265 | 50.7630 | 2.9137 |
| random | n_trial | 32.0000 | 3.1501 | 0.2181 | 0.8956 | 0.0344 | 65.4860 | 1.9353 |
| sampling | baseline | - | - | - | 0.8923 | 0.0313 | 3.2545 | 46.8957 |
| temperature | baseline | - | - | - | 0.9826 | 0.0012 | 3.3548 | 46.0174 |
| top_k | baseline | - | - | - | 0.8525 | 0.0257 | 3.3943 | 41.5149 |
| top_p | baseline | - | - | - | 0.9676 | 0.0034 | 3.5424 | 39.3085 |

## Flatness-N vs Random-N

| n_trials | random | minflat | minflat_minus_random | relative_delta |
|---|---|---|---|---|
| 1.0000 | 3.1478 | 3.1478 | 0.0000 | 0.0000 |
| 4.0000 | 3.1406 | 2.1449 | -0.9957 | -0.3170 |
| 8.0000 | 3.1541 | 1.9155 | -1.2386 | -0.3927 |
| 16.0000 | 3.1515 | 1.8171 | -1.3344 | -0.4234 |
| 32.0000 | 3.1501 | 1.7507 | -1.3994 | -0.4443 |

## Flatness control table

| n_trials | random | minflat | maxflat | minflat_minus_random | maxflat_minus_random | maxflat_minus_minflat |
|---|---|---|---|---|---|---|
| 1.0000 | 3.1478 | 3.1478 | 3.1478 | 0.0000 | 0.0000 | 0.0000 |
| 4.0000 | 3.1406 | 2.1449 | 3.6848 | -0.9957 | 0.5443 | 1.5400 |
| 8.0000 | 3.1541 | 1.9155 | 3.8013 | -1.2386 | 0.6472 | 1.8859 |
| 16.0000 | 3.1515 | 1.8171 | 3.8901 | -1.3344 | 0.7386 | 2.0730 |
| 32.0000 | 3.1501 | 1.7507 | 3.9499 | -1.3994 | 0.7998 | 2.1993 |

## N dependence

| method | n_trials | mean_chosen_flatness | mean_candidate_flatness | mean_selection_gain | mean_candidate_unique_count | distinct_2 | repetition_4 | mean_runtime_seconds |
|---|---|---|---|---|---|---|---|---|
| maxflat | 1.0000 | 3.1478 | 3.1478 | 0.0000 | 1.0000 | 0.9074 | 0.0223 | 9.8593 |
| maxflat | 4.0000 | 3.6848 | 3.3533 | 0.1043 | 3.1058 | 0.8783 | 0.0159 | 15.7482 |
| maxflat | 8.0000 | 3.8013 | 3.3674 | 0.1390 | 5.3337 | 0.8662 | 0.0113 | 22.1860 |
| maxflat | 16.0000 | 3.8901 | 3.3844 | 0.1668 | 9.0706 | 0.8311 | 0.0216 | 32.3772 |
| maxflat | 32.0000 | 3.9499 | 3.3878 | 0.1904 | 15.6215 | 0.7883 | 0.0564 | 49.2001 |
| minflat | 1.0000 | 3.1478 | 3.1478 | 0.0000 | 1.0000 | 0.9074 | 0.0223 | 5.3896 |
| minflat | 4.0000 | 2.1449 | 2.3437 | 0.0807 | 3.2754 | 0.8612 | 0.0900 | 11.4337 |
| minflat | 8.0000 | 1.9155 | 2.1237 | 0.0936 | 6.1264 | 0.8127 | 0.1079 | 18.7473 |
| minflat | 16.0000 | 1.8171 | 2.0358 | 0.1028 | 11.0177 | 0.7953 | 0.1116 | 38.2321 |
| minflat | 32.0000 | 1.7507 | 1.9876 | 0.1154 | 21.0015 | 0.8455 | 0.0670 | 89.5645 |
| random | 1.0000 | 3.1478 | 3.1478 | 0.0000 | 1.0000 | 0.9074 | 0.0223 | 9.2535 |
| random | 4.0000 | 3.1406 | 3.1438 | 0.1101 | 3.2007 | 0.9021 | 0.0279 | 17.3477 |
| random | 8.0000 | 3.1541 | 3.1565 | 0.1516 | 5.6261 | 0.8877 | 0.0373 | 26.8845 |
| random | 16.0000 | 3.1515 | 3.1513 | 0.1875 | 9.8050 | 0.8959 | 0.0265 | 50.7630 |
| random | 32.0000 | 3.1501 | 3.1552 | 0.2181 | 17.1200 | 0.8956 | 0.0344 | 65.4860 |

## Correlation with chosen flatness

| metric | corr_with_mean_chosen_flatness |
|---|---|
| mean_selection_gain | 0.2080 |
| distinct_1 | -0.6479 |
| distinct_2 | -0.1454 |
| distinct_3 | -0.0149 |
| repetition_4 | 0.0222 |
| mean_runtime_seconds | -0.1136 |
| mean_tokens_per_second | -0.0220 |

## Interpretation checklist

- Does `minflat` reduce chosen flatness relative to `random` at the same N?
- Does `maxflat` move in the opposite direction as a negative control?
- Does increasing N change chosen flatness monotonically or approximately?
- Does lower flatness trade off with distinct-n or repetition?
- How quickly does runtime increase as N grows?
