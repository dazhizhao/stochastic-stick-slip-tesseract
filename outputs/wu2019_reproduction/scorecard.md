# Wu2019 vs JAX-FEM

This is a Wu-method reproduction on the frozen 32x4 JAX-FEM/Jenkins benchmark, not a reproduction of the paper's SDOF/MHBM solver.

FAST scope: **Wu-scale resonance-only FAST reproduction** (`N=1000`, 8000 total samples).

## FAST sensitivity

| Rank | Parameter | S1 | ST |
|---:|---|---:|---:|
| 1 | Phi2 | 0.505607 | 0.705236 |
| 2 | Phi4 | 0.167572 | 0.239924 |
| 3 | A2 | 0.00807528 | 0.173762 |
| 4 | A4 | 0.0304151 | 0.0973249 |
| 5 | A3 | 0.00980878 | 0.0305615 |
| 6 | Phi3 | 0.00336045 | 0.030415 |
| 7 | Phi1 | 0.000617151 | 0.0278628 |
| 8 | A1 | 3.70431e-05 | 0.0162079 |

The paper's Fig. 9 uses the maximum response over a frequency band; W1 uses the frozen-resonance amplitude, so the comparison is ordinal rather than a percentage-error comparison.

## Harmonic comparison

| Harmonic | Best A/A0 | Best phase (rad) | Reduction at omega_r (%) | Sampled local max ratio | Sampled local max reduction (%) | Range status |
|---|---:|---:|---:|---:|---:|---|
| 1omega | 0.25 | 5.89049 | 0.687934 | 1 | 0.687934 | interior |
| 2omega | 0.25 | 4.51604 | 21.6068 | 1.04 | 20.2292 | interior |
| 3omega | 0.25 | 0.19635 | 7.85139 | 1.01 | 7.84034 | interior |
| 4omega | 0.25 | 5.20326 | 18.4263 | 1.1 | 12.521 | range_insufficient |

## Friction energy

| Control | Mean dissipated energy/period | Change vs passive (%) |
|---|---:|---:|
| passive | 0.0076793766 | 0 |
| 1omega | 0.0076466564 | -0.426079 |
| 2omega | 0.0064873246 | -15.5228 |
| 4omega | 0.0067283378 | -12.3843 |

## Excitation working range

Discrete active intervals: `[[0.7, 1.8]]`. No interpolation or extrapolation was used.

## Interpretation

**Partial reproduction**

- 2omega_FAST_dominant_over_1omega_3omega: `True`
- 4omega_FAST_secondary_over_1omega_3omega: `True`
- local_peak_order_2_then_4_then_1_3: `True`
- 2omega_local_peak_reduction_is_20_percent_scale: `True`
- active_working_range_extends_below_and_above_design: `True`
- passive_2omega_4omega_local_peak_ranges_sufficient: `False`
- dissipated_energy_order_2omega_then_4omega_then_passive: `False`

## Provisional stochastic appendix

These fixed-bank results are not part of the Wu2019 reproduction.

| Adam lr | Neutral | Final | Improvement (%) |
|---:|---:|---:|---:|
| 0.1 | 0.18171339 | 0.13723819 | 24.4755 |
| 1 | 0.18171339 | 0.13389546 | 26.315 |
