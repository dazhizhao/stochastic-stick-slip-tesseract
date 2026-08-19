"""Probe low-damping, near-resonance operating conditions without training."""

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

from stochastic_stick_slip.model import (
    FORCING_AMPLITUDE,
    NUM_FOURIER_COEFFICIENTS,
    TRAINING_SEEDS,
    build_batch_simulator,
    forcing_parameters,
)
from stochastic_stick_slip.showcase import FOURIER_BASIS, SYSTEM


OUTPUT_PATH = ROOT / "outputs/engineering_probe/probe_results.json"
PROBE_SEEDS = np.concatenate(
    (TRAINING_SEEDS, np.arange(201, 225, dtype=np.int64))
)
OPERATING_CONDITIONS = (
    (0.20, 0.95),
    (0.20, 1.00),
    (0.10, 0.95),
    (0.10, 1.00),
    (0.05, 0.95),
    (0.05, 1.00),
)
BASE_PRELOAD = 0.04
ENDPOINT_PRELOADS = (0.02, 0.06)
SECOND_FREQUENCY_RATIO = 1.35
SCIENTIFIC_REFERENCE_OBJECTIVE = 0.007660674831379117
SHOWCASE_OBJECTIVE_TARGET = 1.25 * SCIENTIFIC_REFERENCE_OBJECTIVE
NUMERICAL_RTOL = 1e-8
NUMERICAL_ATOL = 1e-12


def _forcing_batch(first_frequency_ratio: float) -> jax.Array:
    times = np.asarray(SYSTEM.times)
    histories = []
    for seed in PROBE_SEEDS:
        amplitudes, phases = forcing_parameters(int(seed))
        histories.append(
            FORCING_AMPLITUDE
            * (
                amplitudes[0]
                * np.sin(
                    first_frequency_ratio * SYSTEM.omega_1 * times + phases[0]
                )
                + 0.6
                * amplitudes[1]
                * np.sin(
                    SECOND_FREQUENCY_RATIO * SYSTEM.omega_1 * times + phases[1]
                )
            )
        )
    return jnp.asarray(np.stack(histories), dtype=jnp.float64)


def _summarize_forward(outputs, damping: float, ratio: float, preload: float):
    displacement, velocity, slip, stick_to_slip, slip_to_stick = (
        np.asarray(value) for value in outputs
    )
    losses = np.mean(displacement**2, axis=1)
    visits_both = np.logical_and(
        np.any(slip, axis=1), np.any(np.logical_not(slip), axis=1)
    )
    complete_cycles = np.logical_and(
        np.any(stick_to_slip, axis=1), np.any(slip_to_stick, axis=1)
    )
    finite = bool(
        np.all(np.isfinite(displacement))
        and np.all(np.isfinite(velocity))
        and np.all(np.isfinite(losses))
    )
    mean_objective = float(np.mean(losses))
    summary = {
        "damping": damping,
        "first_frequency_ratio": ratio,
        "preload": preload,
        "mean_objective": mean_objective,
        "observation_rms": float(np.sqrt(mean_objective)),
        "peak_abs_displacement": float(np.max(np.abs(displacement))),
        "stick_to_slip": np.sum(stick_to_slip, axis=(0, 1)).astype(int).tolist(),
        "slip_to_stick": np.sum(slip_to_stick, axis=(0, 1)).astype(int).tolist(),
        "seeds_with_stick_and_slip": {
            "contact_A": int(np.count_nonzero(visits_both[:, 0])),
            "contact_B": int(np.count_nonzero(visits_both[:, 1])),
            "both_contacts": int(np.count_nonzero(np.all(visits_both, axis=1))),
        },
        "seeds_with_complete_cycles": {
            "contact_A": int(np.count_nonzero(complete_cycles[:, 0])),
            "contact_B": int(np.count_nonzero(complete_cycles[:, 1])),
            "both_contacts": int(np.count_nonzero(np.all(complete_cycles, axis=1))),
        },
        "finite": finite,
    }
    if np.isclose(preload, BASE_PRELOAD, rtol=0.0, atol=0.0):
        summary["passes_nominal_selection"] = bool(
            finite
            and summary["seeds_with_complete_cycles"]["both_contacts"] >= 17
            and mean_objective >= SHOWCASE_OBJECTIVE_TARGET
        )
    return summary, displacement


def _run_forward(simulator, damping: float, ratio: float, preload: float):
    coefficients = np.zeros(
        (len(PROBE_SEEDS), NUM_FOURIER_COEFFICIENTS), dtype=np.float64
    )
    outputs = simulator(
        jnp.asarray([damping, preload], dtype=jnp.float64),
        jnp.asarray(coefficients),
        _forcing_batch(ratio),
    )
    jax.block_until_ready(outputs[0])
    return _summarize_forward(outputs, damping, ratio, preload)


def _authority_metrics(endpoint, endpoint_displacement, nominal, nominal_displacement):
    objective_relative_change = (
        (endpoint["mean_objective"] - nominal["mean_objective"])
        / nominal["mean_objective"]
    )
    peak_relative_change = (
        (endpoint["peak_abs_displacement"] - nominal["peak_abs_displacement"])
        / nominal["peak_abs_displacement"]
    )
    trajectory_difference = endpoint_displacement - nominal_displacement
    normalized_trajectory_rms = float(
        np.sqrt(np.mean(trajectory_difference**2))
        / np.sqrt(np.mean(nominal_displacement**2))
    )
    endpoint["objective_relative_change_percent"] = float(
        100.0 * objective_relative_change
    )
    endpoint["peak_relative_change_percent"] = float(100.0 * peak_relative_change)
    endpoint["normalized_trajectory_rms_difference_percent"] = float(
        100.0 * normalized_trajectory_rms
    )
    endpoint["objective_changed"] = bool(
        not np.isclose(
            endpoint["mean_objective"], nominal["mean_objective"],
            rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL,
        )
    )
    endpoint["trajectory_changed"] = bool(
        not np.allclose(
            endpoint_displacement, nominal_displacement,
            rtol=NUMERICAL_RTOL, atol=NUMERICAL_ATOL,
        )
    )
    return endpoint


def main() -> int:
    simulator = build_batch_simulator(SYSTEM, FOURIER_BASIS)
    cases = []
    displacements = []
    for damping, ratio in OPERATING_CONDITIONS:
        result, displacement = _run_forward(
            simulator, damping, ratio, BASE_PRELOAD
        )
        cases.append(result)
        displacements.append(displacement)
        print(
            f"c={damping:.2f} r1={ratio:.2f} "
            f"J={result['mean_objective']:.16g} "
            f"RMS={result['observation_rms']:.9g} "
            f"peak={result['peak_abs_displacement']:.9g} "
            f"cycles_both={result['seeds_with_complete_cycles']['both_contacts']}/32 "
            f"candidate={result['passes_nominal_selection']}",
            flush=True,
        )

    candidate_index = next(
        (index for index, result in enumerate(cases) if result["passes_nominal_selection"]),
        None,
    )
    endpoint_results = []
    selected_case = None
    if candidate_index is None:
        selection_reason = (
            "No nominal case met the predeclared vibration and two-contact "
            "switching conditions."
        )
    else:
        candidate = cases[candidate_index]
        nominal_displacement = displacements[candidate_index]
        for preload in ENDPOINT_PRELOADS:
            endpoint, endpoint_displacement = _run_forward(
                simulator,
                candidate["damping"],
                candidate["first_frequency_ratio"],
                preload,
            )
            endpoint_results.append(
                _authority_metrics(
                    endpoint, endpoint_displacement, candidate, nominal_displacement
                )
            )
        authority_pass = bool(
            all(result["finite"] for result in endpoint_results)
            and any(
                result["objective_changed"] or result["trajectory_changed"]
                for result in endpoint_results
            )
        )
        if authority_pass:
            selected_case = {
                key: candidate[key]
                for key in (
                    "damping", "first_frequency_ratio", "preload",
                    "mean_objective", "observation_rms", "peak_abs_displacement",
                    "stick_to_slip", "slip_to_stick",
                    "seeds_with_stick_and_slip", "seeds_with_complete_cycles",
                )
            }
            selection_reason = (
                "First nominal case in the fixed priority order meeting the "
                "showcase vibration and switching conditions; both preload "
                "endpoints were finite and changed the response."
            )
        else:
            selection_reason = (
                "The first nominal candidate failed the bounded preload-authority "
                "check; no later candidate was tested."
            )

    output = {
        "scientific_reference": {
            "mean_objective": SCIENTIFIC_REFERENCE_OBJECTIVE,
            "observation_rms": float(np.sqrt(SCIENTIFIC_REFERENCE_OBJECTIVE)),
            "showcase_objective_target": SHOWCASE_OBJECTIVE_TARGET,
        },
        "cases": cases,
        "endpoint_checks": endpoint_results,
        "selected_case": selected_case,
        "selection_reason": selection_reason,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n")

    print("## Selection")
    print(selection_reason)
    for endpoint in endpoint_results:
        print(
            f"N={endpoint['preload']:.2f} "
            f"J_change={endpoint['objective_relative_change_percent']:.6f}% "
            f"peak_change={endpoint['peak_relative_change_percent']:.6f}% "
            f"trajectory_delta={endpoint['normalized_trajectory_rms_difference_percent']:.6f}%"
        )
    print(OUTPUT_PATH.resolve())
    return 0 if selected_case is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
