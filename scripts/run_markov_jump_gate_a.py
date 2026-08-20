"""Run the locked Markov-jump mechanics preflight and Gate A."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from stochastic_stick_slip.engineering_markov import (
    BETA,
    DAMPING,
    FD_EPSILON,
    GATE_A_FORCING_SEEDS,
    LAMBDA_0,
    MARKOV_BASE_SEED,
    MARKOV_ITERATION,
    MARKOV_STREAM,
    MECHANICS_SIMULATOR,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    SYSTEM,
    direct_ad_objective_and_gradient,
    evaluate_markov_bank,
    gate_a_forcing,
    markov_uniform_bank,
)
from stochastic_stick_slip.engineering_showcase import _SIMULATE_BATCH
from stochastic_stick_slip.model import NUM_FOURIER_COEFFICIENTS, NUM_STEPS


OUTPUT_PATH = ROOT / "outputs/markov_jump_gate_a/results.json"
NUMERICAL_ZERO_ATOL = 1e-12
FIXED_PRELOAD_CASES = {
    "LL": (PRELOAD_LOW, PRELOAD_LOW),
    "LH": (PRELOAD_LOW, PRELOAD_HIGH),
    "HL": (PRELOAD_HIGH, PRELOAD_LOW),
    "HH": (PRELOAD_HIGH, PRELOAD_HIGH),
}


def _as_arrays(outputs):
    return tuple(np.asarray(output) for output in outputs)


def _fixed_preload_preflight(forcing):
    zero_coefficients = np.zeros(
        (len(GATE_A_FORCING_SEEDS), NUM_FOURIER_COEFFICIENTS),
        dtype=np.float64,
    )
    summaries = {}
    for name, contact_preloads in FIXED_PRELOAD_CASES.items():
        preload = np.broadcast_to(
            np.asarray(contact_preloads, dtype=np.float64),
            (len(GATE_A_FORCING_SEEDS), NUM_STEPS, 2),
        )
        outputs = _as_arrays(
            MECHANICS_SIMULATOR(
                jnp.asarray(DAMPING, dtype=jnp.float64),
                forcing,
                jnp.asarray(preload),
            )
        )
        displacement, velocity, _, stick_to_slip, slip_to_stick = outputs
        finite = bool(
            np.all(np.isfinite(displacement))
            and np.all(np.isfinite(velocity))
        )
        legacy_aligned = None
        if name in ("LL", "HH"):
            legacy = _as_arrays(
                _SIMULATE_BATCH(
                    jnp.asarray(
                        [DAMPING, contact_preloads[0]], dtype=jnp.float64
                    ),
                    jnp.asarray(zero_coefficients),
                    forcing,
                )
            )
            legacy_aligned = bool(
                all(
                    np.allclose(
                        candidate,
                        reference,
                        rtol=1e-12,
                        atol=1e-14,
                    )
                    for candidate, reference in zip(outputs, legacy, strict=True)
                )
            )
        losses = np.mean(displacement**2, axis=1)
        summaries[name] = {
            "preload": list(contact_preloads),
            "finite": finite,
            "legacy_aligned": legacy_aligned,
            "seed_losses": losses.tolist(),
            "mean_objective": float(np.mean(losses)),
            "stick_to_slip": np.sum(stick_to_slip, axis=(0, 1)).astype(int).tolist(),
            "slip_to_stick": np.sum(slip_to_stick, axis=(0, 1)).astype(int).tolist(),
        }
        if not finite:
            raise FloatingPointError(f"{name} fixed-preload forward is non-finite")
        if legacy_aligned is False:
            raise AssertionError(f"{name} does not reproduce the legacy forward")
    return summaries


def _neutral_summary(result):
    losses = np.asarray(result.losses)
    return {
        "objective": float(np.mean(losses)),
        "condition_losses": np.mean(losses, axis=1).tolist(),
        "transition_counts": np.asarray(result.transition_counts).astype(int).tolist(),
        "high_mode_fraction": np.asarray(result.high_mode_fraction).tolist(),
        "stick_to_slip": np.asarray(result.stick_to_slip).astype(int).tolist(),
        "slip_to_stick": np.asarray(result.slip_to_stick).astype(int).tolist(),
    }


def _gradient_summary(gradient):
    gradient = np.asarray(gradient, dtype=np.float64)
    finite = bool(np.all(np.isfinite(gradient)))
    if not finite:
        raise FloatingPointError("coefficient gradient is non-finite")
    linf = float(np.max(np.abs(gradient)))
    return {
        "gradient": gradient.tolist(),
        "l2_norm": float(np.linalg.norm(gradient)),
        "linf_norm": linf,
        "finite": finite,
        "exact_zero": bool(np.array_equal(gradient, np.zeros_like(gradient))),
        "numerical_zero": bool(linf <= NUMERICAL_ZERO_ATOL),
    }


def _centered_fd(shared_coefficients, forcing, uniforms):
    gradients = []
    plus_objectives = []
    minus_objectives = []
    mode_difference_counts = []
    jenkins_difference_counts = []
    num_conditions = forcing.shape[0]

    for column in range(NUM_FOURIER_COEFFICIENTS):
        plus = shared_coefficients.copy()
        minus = shared_coefficients.copy()
        plus[column] += FD_EPSILON
        minus[column] -= FD_EPSILON
        plus_coefficients = np.broadcast_to(
            plus, (num_conditions, NUM_FOURIER_COEFFICIENTS)
        )
        minus_coefficients = np.broadcast_to(
            minus, (num_conditions, NUM_FOURIER_COEFFICIENTS)
        )
        plus_result = evaluate_markov_bank(
            plus_coefficients, forcing, uniforms
        )
        minus_result = evaluate_markov_bank(
            minus_coefficients, forcing, uniforms
        )
        plus_objective = float(np.mean(np.asarray(plus_result.losses)))
        minus_objective = float(np.mean(np.asarray(minus_result.losses)))
        gradients.append(
            (plus_objective - minus_objective) / (2.0 * FD_EPSILON)
        )
        plus_objectives.append(plus_objective)
        minus_objectives.append(minus_objective)
        mode_difference_counts.append(
            int(
                np.count_nonzero(
                    np.any(
                        np.asarray(plus_result.modes)
                        != np.asarray(minus_result.modes),
                        axis=(2, 3),
                    )
                )
            )
        )
        jenkins_difference_counts.append(
            int(
                np.count_nonzero(
                    np.any(
                        np.asarray(plus_result.slip)
                        != np.asarray(minus_result.slip),
                        axis=(2, 3),
                    )
                )
            )
        )

    summary = _gradient_summary(np.asarray(gradients))
    summary.update(
        {
            "epsilon": FD_EPSILON,
            "plus_objectives": plus_objectives,
            "minus_objectives": minus_objectives,
            "mode_difference_counts": mode_difference_counts,
            "jenkins_difference_counts": jenkins_difference_counts,
        }
    )
    return summary


def _run_attempt(num_realizations, forcing, shared_coefficients):
    uniforms = markov_uniform_bank(num_realizations)
    coefficients = np.broadcast_to(
        shared_coefficients,
        (len(GATE_A_FORCING_SEEDS), NUM_FOURIER_COEFFICIENTS),
    )
    neutral = evaluate_markov_bank(coefficients, forcing, uniforms)
    if not np.all(np.isfinite(np.asarray(neutral.losses))):
        raise FloatingPointError("neutral Markov objective is non-finite")
    objective, direct_gradient = direct_ad_objective_and_gradient(
        shared_coefficients, forcing, uniforms
    )
    direct_ad = _gradient_summary(direct_gradient)
    direct_ad["objective"] = objective
    if not direct_ad["numerical_zero"]:
        return _neutral_summary(neutral), direct_ad, None
    return (
        _neutral_summary(neutral),
        direct_ad,
        _centered_fd(shared_coefficients, forcing, uniforms),
    )


def _write_results(results):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")


def _print_results(results):
    print("## Mechanics refactor")
    for name, summary in results["preflight"].items():
        print(
            f"{name}: J={summary['mean_objective']:.16g} "
            f"finite={summary['finite']} "
            f"legacy_aligned={summary['legacy_aligned']} "
            f"stick_to_slip={summary['stick_to_slip']} "
            f"slip_to_stick={summary['slip_to_stick']}"
        )
    print("## Markov process")
    print(f"realizations={results['num_realizations']}")
    print(f"neutral_objective={results['neutral_markov']['objective']:.16g}")
    print(
        "transition_counts="
        f"{results['neutral_markov']['transition_counts']}"
    )
    print(
        "high_mode_fraction="
        f"{results['neutral_markov']['high_mode_fraction']}"
    )
    print("## Direct AD")
    direct_ad = results["direct_ad"]
    print(f"g_AD={direct_ad['gradient']}")
    print(f"L2={direct_ad['l2_norm']:.16g}")
    print(f"Linf={direct_ad['linf_norm']:.16g}")
    print(
        f"finite={direct_ad['finite']} exact_zero={direct_ad['exact_zero']} "
        f"numerical_zero={direct_ad['numerical_zero']}"
    )
    print("## CRN-FD")
    crn_fd = results["crn_fd"]
    if crn_fd is None:
        print("skipped: Direct AD was non-zero")
    else:
        print(f"g_FD={crn_fd['gradient']}")
        print(f"L2={crn_fd['l2_norm']:.16g}")
        print(f"Linf={crn_fd['linf_norm']:.16g}")
        print(f"mode_difference_counts={crn_fd['mode_difference_counts']}")
        print(
            "jenkins_difference_counts="
            f"{crn_fd['jenkins_difference_counts']}"
        )
    print("## Gate A")
    print(f"Gate A: {results['gate_a']['result']}")
    print(f"reason={results['gate_a']['reason']}")
    print(OUTPUT_PATH.resolve())


def main() -> int:
    forcing = gate_a_forcing()
    preflight = _fixed_preload_preflight(forcing)
    shared_coefficients = np.zeros(NUM_FOURIER_COEFFICIENTS, dtype=np.float64)

    num_realizations = 4
    neutral, direct_ad, crn_fd = _run_attempt(
        num_realizations, forcing, shared_coefficients
    )
    fallback_from_r4 = False
    if crn_fd is not None and crn_fd["numerical_zero"]:
        fallback_from_r4 = True
        num_realizations = 8
        neutral, direct_ad, crn_fd = _run_attempt(
            num_realizations, forcing, shared_coefficients
        )

    ad_pass = direct_ad["numerical_zero"]
    fd_finite = crn_fd is not None and crn_fd["finite"]
    fd_nonzero = crn_fd is not None and not crn_fd["numerical_zero"]
    mode_changed = bool(
        crn_fd is not None and any(crn_fd["mode_difference_counts"])
    )
    passed = bool(ad_pass and fd_finite and fd_nonzero and mode_changed)
    if not ad_pass:
        reason = "Direct AD is non-zero; audit for a continuous shortcut."
    elif not fd_nonzero:
        reason = "CRN-FD remained numerical-zero after the allowed R=8 fallback."
    elif not mode_changed:
        reason = "CRN-FD perturbations did not change any Markov mode history."
    else:
        reason = "Direct AD is zero while finite non-zero CRN-FD changes hard modes."

    results = {
        "configuration": {
            "forcing_seeds": GATE_A_FORCING_SEEDS.tolist(),
            "markov_base_seed": MARKOV_BASE_SEED,
            "markov_stream": MARKOV_STREAM,
            "markov_iteration": MARKOV_ITERATION,
            "uniform_shape": [
                len(GATE_A_FORCING_SEEDS),
                num_realizations,
                NUM_STEPS + 1,
                2,
            ],
            "damping": DAMPING,
            "preload_low": PRELOAD_LOW,
            "preload_high": PRELOAD_HIGH,
            "lambda_0": LAMBDA_0,
            "beta": BETA,
            "fd_epsilon": FD_EPSILON,
            "numerical_zero_atol": NUMERICAL_ZERO_ATOL,
        },
        "preflight": preflight,
        "num_realizations": num_realizations,
        "fallback_from_r4": fallback_from_r4,
        "neutral_markov": neutral,
        "direct_ad": direct_ad,
        "crn_fd": crn_fd,
        "gate_a": {"result": "PASS" if passed else "FAIL", "reason": reason},
    }
    _write_results(results)
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
