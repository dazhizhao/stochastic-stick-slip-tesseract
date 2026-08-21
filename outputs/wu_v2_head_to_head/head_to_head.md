# Wu-V2 stochastic Markov vs Wu-style deterministic 2ω

All stochastic results use four new independent confirmation banks (64 Markov realizations per bank, 256 total).

## Head-to-head scorecard

| Method | Nominal response | Local peak | vs Passive | vs Wu 2ω |
|---|---:|---:|---:|---:|
| Passive | 0.187487205 | 0.187487205 | 0.0000% | -25.3592% |
| Wu-style continuous 2ω | 0.146977152 | 0.149559977 | 20.2292% | — |
| Deterministic binary 2ω | 0.143042129 | 0.149130232 | 20.4584% | 0.2873% |
| Stochastic, lr=0.1 | 0.137640571 | 0.144729497 | 22.8057% | 3.2298% |
| Stochastic, lr=1.0 | 0.133881709 | 0.143036275 | 23.7088% | 4.3619% |

## Independent-bank sampled peaks

| Policy | Aggregate | Stream 5 | Stream 6 | Stream 7 | Stream 8 |
|---|---:|---:|---:|---:|---:|
| Stochastic, lr=0.1 | 0.144729497 | 0.144553671 | 0.144836208 | 0.144775687 | 0.144752421 |
| Stochastic, lr=1.0 | 0.143036275 | 0.143036400 | 0.143047985 | 0.143001904 | 0.143058812 |

## Markov policy diagnostics

### Stochastic, lr=0.1

- q: `[-3.514982271973227, -0.5337623522652377]`
- magnitude: `3.555278136`
- coefficient phase: `3.292294818 rad`
- mean HIGH occupancy/contact: `[0.4992122395833334, 0.49906412760416674]`
- mean transitions/trajectory/contact: `[104.66796875, 105.01171875]`
- realizations below the Wu local peak at the stochastic aggregate peak: `256/256`

### Stochastic, lr=1.0

- q: `[-10.665739565561044, -6.033414703985564]`
- magnitude: `12.253982760`
- coefficient phase: `3.656395859 rad`
- mean HIGH occupancy/contact: `[0.5002718098958332, 0.5001725260416664]`
- mean transitions/trajectory/contact: `[96.90625, 96.9609375]`
- realizations below the Wu local peak at the stochastic aggregate peak: `256/256`

## Conclusion

The optimized stochastic Markov policy outperforms the Wu-style deterministic 2ω sinusoidal control on the same JAX-FEM benchmark.

This compares implementations on the same JAX-FEM benchmark; it does not claim to outperform Wu et al. 2019.
