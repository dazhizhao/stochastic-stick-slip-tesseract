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

Run the reproduction with:

```bash
uv run python experiments/wu2019/scripts/reproduce_wu2019.py
uv run python experiments/wu2019/scripts/optimize_markov.py
uv run pytest -q experiments/wu2019/tests
```

Generated numerical summaries and PNG figures are written to the local,
git-ignored `experiments/wu2019/outputs/` directory.
