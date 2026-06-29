# Wrapped10 vs Wrapped20

Wrapped20 is the better audit set. Wrapped10 is useful as a stability check, but the qualitative conclusions should be anchored on the 20-prompt run.

| method | rows10 | rows20 | rain10 | rain20 | generic_atmo10 | generic_atmo20 | prompt_reuse10 | prompt_reuse20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| top_p | 10 | 20 | 0.6 | 0.65 | 0.0 | 0.0 | 0.2 | 0.3 |
| top_k | 10 | 20 | 0.7 | 0.7 | 0.0 | 0.0 | 0.3 | 0.35 |
| random | 10 | 20 | 0.6 | 0.7 | 0.0 | 0.0 | 0.1 | 0.25 |
| minflat | 10 | 20 | 0.4 | 0.35 | 0.0 | 0.0 | 0.0 | 0.2 |
| maxflat | 10 | 20 | 0.7 | 0.55 | 0.0 | 0.0 | 0.1 | 0.15 |
| centerflat_q50 | 10 | 20 | 0.8 | 0.8 | 0.0 | 0.0 | 0.1 | 0.35 |
| centerflat_q75 | 10 | 20 | 0.8 | 0.75 | 0.0 | 0.0 | 0.0 | 0.1 |
| highprob_centerflat | 10 | 20 | 0.8 | 0.8 | 0.0 | 0.0 | 0.0 | 0.1 |

The signal that strengthens from 10 to 20 is not robust format control. It is a more stable estimate of a residual prose prior: methods stay in direct continuation mode, but many outputs still start with rain, metal, rust, regret, or other atmosphere-heavy language.
