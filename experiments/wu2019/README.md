# Wu 2019 External Benchmark Prototype

This isolated experiment reproduces the numerical SDOF benchmark from:

Y. G. Wu et al., “Design of semi-active dry friction dampers for
steady-state vibration: sensitivity analysis and experimental studies,”
*Journal of Sound and Vibration* 459 (2019) 114850.

The experiment is deliberately separate from the repository's existing
JAX-FEM/Tesseract benchmark. It contains the dimensional SDOF model, hard
stick/slip contact, Newmark integration, and a standalone two-state Markov
controller. It does not import the existing mechanics or Tesseract components.

Published semi-active friction dampers optimize a continuous normal-force
waveform. We ask whether a hard two-state stochastic actuator, whose gradient
is unavailable to standard autodiff, can recover or surpass this performance
using mixed gradients.

## Phase 1: published benchmark reproduction

The fixed-parameter reproduction passes its scientific gates:

| Controller | Peak response | Reduction vs N=40 |
|---|---:|---:|
| Constant N=40 | 2.386861 mm | 0% |
| Wu 2ω continuous | 1.869376 mm | 21.6806% |
| Wu 4ω continuous | 2.152013 mm | 9.8392% |

The constant-preload optimum is 40 N. The near-best 1ω sanity check gives a
3.4101% reduction: this is retained as a diagnostic because the paper describes
the first harmonic qualitatively as having little effect, without specifying a
3% numerical limit.

## Phase 2: hard Markov result

The hard controller switches only between 30 N and 50 N. A shared five-term
Fourier signal controls the two transition rates, and its coefficients are
optimized with same-tape centered CRN finite differences and Adam.

| Controller | Peak response | Reduction vs N=40 |
|---|---:|---:|
| Constant N=40 | 2.386861 mm | 0% |
| Wu continuous | 1.869376 mm | 21.6806% |
| Hard Markov initial | 2.401685 mm | -0.6211% |
| Hard Markov optimized | 2.364171 mm | 0.9506% |

The optimized hard controller improves 1.5620% relative to its own initial
state, but remains 20.7300 percentage points behind the Wu continuous
controller. It does not meet the 18% science gate for a Tesseract phase.

## Phase 2B: causal state-aware hard Markov result

Phase 2B keeps the frozen Phase 2 actuator, stochastic tapes, frequency grids,
CRN-FD step, Adam schedule, and objective. It adds only two feedback gains. At
step `n`, the switching score uses the displacement and velocity from step
`n-1`:

```text
s_n = Fourier(theta_n) + c_v v_(n-1) / 0.48 + c_x x_(n-1) / 0.0024
```

The normalized state is not clipped. The causal update is previous mechanics
state, hard Markov transition, 30/50 N preload selection, then one Newmark
step. Zero feedback gain reproduces the frozen Phase 2 objective within
`5.5e-16 m` and the complete amplitude bank within `2.4e-15 m`.

The frozen independent `R=32`, `0.25 rad/s` evaluation gives:

| Controller | Peak response | Reduction vs N=40 |
|---|---:|---:|
| Constant N=40 | 2.386861 mm | 0% |
| Wu continuous | 1.869376 mm | 21.6806% |
| Phase 2 periodic Markov | 2.364171 mm | 0.9506% |
| Phase 2B state-aware Markov | 2.301235 mm | 3.5874% |

The optimized seven coefficients are
`[0.1692686194, 0.1491125048, 0.3876661414, -0.9368056271,
1.0636146099, 0.1377500304, -0.0705340412]`. State awareness improves the
independent objective by 2.6621% relative to periodic Markov and raises the
reduction by 2.6368 percentage points. This exceeds the predefined 2-point
material-helpfulness threshold, so state information is useful under the
frozen experiment. The final reduction is nevertheless only 3.5874%, far
below Wu's 21.6806% and the 18% future Tesseract gate. State feedback therefore
helps but is insufficient to remove the performance bottleneck; this result
does not establish actuator discretization as its unique cause.

Raw Direct AD remains exactly zero. The initial seven-dimensional CRN-FD
gradient has L2 norm `3.4928701e-4`; both state gains have nonzero components,
and their perturbations change hard switching histories. Two complete runs
produce identical numerical JSON after excluding the runtime field and
byte-identical PNG figures.

Run the reproduction with:

```bash
uv run python experiments/wu2019/scripts/reproduce_wu2019.py
uv run python experiments/wu2019/scripts/optimize_markov.py
uv run python experiments/wu2019/scripts/optimize_state_aware_markov.py
uv run pytest -q experiments/wu2019/tests
```

Generated numerical summaries and PNG figures are written to the local,
git-ignored `experiments/wu2019/outputs/` directory.
