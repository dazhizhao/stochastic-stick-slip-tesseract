# JumpGrad end-to-end result

**J1 PASS**

## Mixed-gradient evidence

| Gradient route | Norm | Gate |
|---|---:|---:|
| Direct AD physics | 0 (L∞) | True |
| CRN-FD physics | 0.67945193308 (L2) | True |
| End-to-end theta | 1.33983069881 (L2) | True |

## Training

- Fixed monitor, iteration 0: `0.965518406323`
- Fixed monitor, iteration 100: `0.776349658214`
- Condition-dependent q: `True`

## Mean normalized response

| Split | Passive | Wu-style 2ω | Fixed q | JumpGrad |
|---|---:|---:|---:|---:|
| Training | 1.000000000 | 0.829348105 | 0.775432101 | 0.776349658 |
| Held Out | 1.000000000 | 0.841086908 | 0.785711713 | 0.789550024 |

Held-out rankings are reported as observed and are not J1 gates.
