# Phase-optimized deterministic Wu-V2 binary comparator

All stochastic amplitudes are frozen from W2; W3 performs only deterministic forward evaluations.

## Comparison

| Method | Local peak | vs Passive | vs Wu continuous |
|---|---:|---:|---:|
| Passive | 0.187487205118 | 0.000000% | -25.359210% |
| Wu continuous 2ω | 0.149559976847 | 20.229236% | — |
| Wu-phase binary | 0.149130231661 | 20.458449% | 0.287340% |
| Phase-optimized binary | 0.142535675753 | 23.975785% | 4.696645% |
| Learned hard limit, lr=0.1 | 0.151077371377 | 19.419903% | -1.014573% |
| Stochastic lr=0.1 | 0.144729496749 | 22.805667% | 3.229795% |
| Learned hard limit, lr=1.0 | 0.145582187932 | 22.350868% | 2.659661% |
| Stochastic lr=1.0 | 0.143036275056 | 23.708781% | 4.361930% |

## Signed margins

- optimized_binary_minus_stochastic_lr1p0: `-0.000500599303259` (`-0.351210%`), very close within one percent
- hard_limit_lr0p1_minus_stochastic_lr0p1: `0.00634787462782` (`4.201738%`), stochastic lower
- hard_limit_lr1p0_minus_stochastic_lr1p0: `0.00254591287542` (`1.748780%`), stochastic lower

## Interpretation

**Case C**

The phase-optimized deterministic binary controller is lower than both frozen stochastic policies on this benchmark. The optimized-binary and stochastic lr=1.0 peaks are within 1% and therefore very close.

This is a comparator result on the same JAX-FEM benchmark, not a claim that randomness is universally beneficial or that Wu et al. 2019 is outperformed.
