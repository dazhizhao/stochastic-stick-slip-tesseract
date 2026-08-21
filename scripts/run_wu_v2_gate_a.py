"""Run Wu-V2 stochastic 2-omega hard-Markov Gate A."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    SYSTEM,
    single_tone_forcing,
)
from stochastic_stick_slip.wu_v2_markov import (
    CONDITION_LABELS,
    FD_EPSILON,
    MARKOV_BASE_SEED,
    NUM_CONDITIONS,
    NUM_REALIZATIONS,
    NUM_STEPS,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    crn_centered_fd,
    direct_ad_objective_and_gradient,
    evaluate_markov_bank,
    fixed_history_objective,
    markov_uniform_bank,
)


GATE_0_PATH = ROOT / "outputs/wu_v2_gate0_final/results.json"
OUTPUT_PATH = ROOT / "outputs/wu_v2_gate_a/results.json"
Q0 = np.zeros(2, dtype=np.float64)
NUMERICAL_ZERO_ATOL = 1e-12
REPLAY_RTOL = 1e-12
REPLAY_ATOL = 1e-14


def _load_wu_references() -> tuple[float, float, dict]:
    gate_0 = json.loads(GATE_0_PATH.read_text())
    if gate_0["gate_0"]["result"] != "PASS":
        raise RuntimeError("Final Wu-style Gate 0 is not PASS")
    configuration = gate_0["configuration"]
    preload = float(configuration["frozen_preload"])
    omega_ratio = float(gate_0["passive"]["omega_r_ratio"])
    omega = float(gate_0["passive"]["omega_r"])
    if not np.isclose(preload, 0.04, rtol=0.0, atol=1e-15):
        raise RuntimeError("Final Gate 0 did not freeze N*=0.04")
    if not np.isclose(omega_ratio, 1.190, rtol=0.0, atol=1e-15):
        raise RuntimeError("Final Gate 0 did not freeze omega_r/omega_1=1.190")
    references = {
        "source": str(GATE_0_PATH.relative_to(ROOT)),
        "passive": float(gate_0["passive"]["steady_amplitude"]),
        "best_deterministic_1omega": float(
            gate_0["phase_sweep"]["one_omega"]["best_amplitude"]
        ),
        "best_deterministic_2omega": float(
            gate_0["phase_sweep"]["two_omega"]["best_amplitude"]
        ),
        "best_deterministic_2omega_local_peak": float(
            gate_0["local_frf"]["two_omega"]["peak_amplitude"]
        ),
    }
    if not all(np.isfinite(value) for value in list(references.values())[1:]):
        raise RuntimeError("Final Gate 0 contains a non-finite Wu reference")
    return omega, omega_ratio, references


def _write_results(results: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")


def _finish(results: dict, result: str, reason: str) -> int:
    results["gate_a"] = {"result": result, "reason": reason}
    _write_results(results)
    print("## Wu references")
    for name, value in results["wu_references"].items():
        print(f"{name}={value}")
    neutral = results["neutral"]
    if neutral is not None:
        print("## Neutral stochastic actuator")
        print(f"mean_amplitude={neutral['mean_amplitude']:.16g}")
        print(
            f"objective_range=[{neutral['minimum_amplitude']:.16g}, "
            f"{neutral['maximum_amplitude']:.16g}]"
        )
        print(f"high_occupancy={neutral['high_occupancy_per_contact']}")
        print(
            "transition_total_per_contact="
            f"{neutral['transition_total_per_contact']}"
        )
    direct_ad = results["direct_ad"]
    if direct_ad is not None:
        print("## Direct AD")
        print(f"g_AD={direct_ad['gradient']}")
        print(f"L2={direct_ad['l2_norm']:.16g}")
        print(f"Linf={direct_ad['linf_norm']:.16g}")
    crn_fd = results["crn_fd"]
    if crn_fd is not None:
        print("## CRN-FD")
        print(f"g_FD={crn_fd['gradient']}")
        print(f"L2={crn_fd['l2_norm']:.16g}")
        print(
            "mode_difference_counts="
            f"{crn_fd['mode_difference_counts']}"
        )
    print("## Gate A")
    print(f"Gate A: {result}")
    print(f"reason={reason}")
    print(OUTPUT_PATH.resolve())
    return 0


def _neutral_summary(evaluation: dict) -> tuple[dict, bool]:
    objectives = np.asarray(evaluation["trajectory_objectives"])
    displacement = np.asarray(evaluation["displacement"])
    velocity = np.asarray(evaluation["velocity"])
    preload = np.asarray(evaluation["preload"])
    modes = np.asarray(evaluation["modes"])
    transition_counts = np.asarray(evaluation["transition_counts"])
    high_occupancy = np.mean(modes, axis=(0, 1, 2), dtype=np.float64)
    low_occupancy = 1.0 - high_occupancy
    transition_totals = np.sum(transition_counts, axis=(0, 1))
    stick_to_slip = np.sum(
        np.asarray(evaluation["stick_to_slip"]), axis=2
    )
    slip_to_stick = np.sum(
        np.asarray(evaluation["slip_to_stick"]), axis=2
    )
    finite = bool(
        np.all(np.isfinite(objectives))
        and np.all(np.isfinite(displacement))
        and np.all(np.isfinite(velocity))
        and np.all(np.isfinite(preload))
    )
    both_modes = bool(
        np.all(high_occupancy > 0.0) and np.all(low_occupancy > 0.0)
    )
    switching = bool(np.all(transition_totals > 0))
    summary = {
        "trajectory_amplitudes": objectives.tolist(),
        "mean_amplitude": float(np.mean(objectives)),
        "minimum_amplitude": float(np.min(objectives)),
        "maximum_amplitude": float(np.max(objectives)),
        "mean_preload": float(np.mean(preload)),
        "low_occupancy_per_contact": low_occupancy.tolist(),
        "high_occupancy_per_contact": high_occupancy.tolist(),
        "transition_counts": transition_counts.astype(int).tolist(),
        "transition_total_per_contact": transition_totals.astype(int).tolist(),
        "jenkins_stick_to_slip": stick_to_slip.astype(int).tolist(),
        "jenkins_slip_to_stick": slip_to_stick.astype(int).tolist(),
        "jenkins_stick_to_slip_total_per_contact": np.sum(
            stick_to_slip, axis=(0, 1)
        ).astype(int).tolist(),
        "jenkins_slip_to_stick_total_per_contact": np.sum(
            slip_to_stick, axis=(0, 1)
        ).astype(int).tolist(),
        "finite": finite,
        "both_modes_present_per_contact": both_modes,
        "switching_present_per_contact": switching,
    }
    return summary, bool(finite and both_modes and switching)


def _gradient_summary(gradient: np.ndarray) -> dict:
    gradient = np.asarray(gradient, dtype=np.float64)
    finite = bool(np.all(np.isfinite(gradient)))
    l2_norm = float(np.linalg.norm(gradient))
    linf_norm = float(np.max(np.abs(gradient)))
    return {
        "gradient": gradient.tolist(),
        "l2_norm": l2_norm,
        "linf_norm": linf_norm,
        "finite": finite,
        "exact_zero": bool(np.array_equal(gradient, np.zeros_like(gradient))),
        "numerical_zero": bool(finite and linf_norm <= NUMERICAL_ZERO_ATOL),
    }


def _fixed_mode_replay(
    forcing: np.ndarray, preload: np.ndarray, time_step: float
) -> dict:
    probes = [
        Q0.copy(),
        np.asarray([FD_EPSILON, 0.0]),
        np.asarray([-FD_EPSILON, 0.0]),
        np.asarray([0.0, FD_EPSILON]),
        np.asarray([0.0, -FD_EPSILON]),
    ]
    objectives = [
        fixed_history_objective(forcing, preload, time_step) for _ in probes
    ]
    reference = objectives[0]
    passed = bool(
        all(
            np.isclose(value, reference, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
            for value in objectives[1:]
        )
    )
    return {
        "trajectory_index": [0, 0],
        "q_probes": [probe.tolist() for probe in probes],
        "objectives": objectives,
        "maximum_absolute_difference": float(
            np.max(np.abs(np.asarray(objectives) - reference))
        ),
        "rtol": REPLAY_RTOL,
        "atol": REPLAY_ATOL,
        "passed": passed,
    }


def main() -> int:
    omega, omega_ratio, references = _load_wu_references()
    time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE, omega, DIAGNOSTIC_NUM_PERIODS
    )
    times = time_step * np.arange(1, NUM_STEPS + 1, dtype=np.float64)
    uniforms = markov_uniform_bank()
    period = 2.0 * np.pi / omega
    results = {
        "configuration": {
            "mesh": "32x4 QUAD4",
            "num_free_dofs": SYSTEM.num_free_dofs,
            "damping": DAMPING,
            "forcing_amplitude": FORCING_AMPLITUDE,
            "omega_r": omega,
            "omega_r_ratio": omega_ratio,
            "period": period,
            "time_step": time_step,
            "num_periods": DIAGNOSTIC_NUM_PERIODS,
            "steps_per_period": NUM_STEPS // DIAGNOSTIC_NUM_PERIODS,
            "q0": Q0.tolist(),
            "preload_low": PRELOAD_LOW,
            "preload_high": PRELOAD_HIGH,
            "lambda_0": 4.0 / period,
            "fd_epsilon": FD_EPSILON,
            "markov_base_seed": MARKOV_BASE_SEED,
            "condition_labels": CONDITION_LABELS.tolist(),
            "num_conditions": NUM_CONDITIONS,
            "num_realizations": NUM_REALIZATIONS,
            "uniform_shape": list(uniforms.shape),
            "forcing_is_identical_for_all_trajectories": True,
            "direct_ad_linf_tolerance": NUMERICAL_ZERO_ATOL,
        },
        "wu_references": references,
        "neutral": None,
        "direct_ad": None,
        "fixed_mode_replay": None,
        "crn_fd": None,
        "gates": {
            "direct_ad_numerical_zero": None,
            "crn_fd_finite": None,
            "crn_fd_nonzero": None,
            "hard_mode_history_changed": None,
        },
        "gate_a": None,
    }

    neutral_evaluation = evaluate_markov_bank(
        Q0, forcing, uniforms, times, omega, time_step
    )
    neutral, neutral_valid = _neutral_summary(neutral_evaluation)
    results["neutral"] = neutral
    if not neutral_valid:
        return _finish(
            results,
            "FAIL",
            "Neutral actuator was non-finite, missed a mode, or did not switch.",
        )

    direct_value, direct_gradient = direct_ad_objective_and_gradient(
        Q0, forcing, uniforms, times, omega, time_step
    )
    direct_ad = _gradient_summary(direct_gradient)
    direct_ad["objective"] = direct_value
    direct_ad["matches_neutral_objective"] = bool(
        np.isclose(
            direct_value,
            neutral["mean_amplitude"],
            rtol=REPLAY_RTOL,
            atol=REPLAY_ATOL,
        )
    )
    results["direct_ad"] = direct_ad
    results["gates"]["direct_ad_numerical_zero"] = direct_ad[
        "numerical_zero"
    ]

    replay = _fixed_mode_replay(
        forcing,
        np.asarray(neutral_evaluation["preload"])[0, 0],
        time_step,
    )
    results["fixed_mode_replay"] = replay
    if not replay["passed"]:
        return _finish(
            results,
            "FAIL",
            "Fixed-mode replay changed across coefficient probes.",
        )
    if not direct_ad["numerical_zero"]:
        return _finish(
            results,
            "FAIL",
            "Direct AD is non-zero; a continuous shortcut remains.",
        )

    fd_outputs = crn_centered_fd(
        Q0, forcing, uniforms, times, omega, time_step
    )
    crn_fd = _gradient_summary(np.asarray(fd_outputs["gradient"]))
    crn_fd.update(
        {
            "epsilon": FD_EPSILON,
            "plus_objectives": fd_outputs["plus_objectives"],
            "minus_objectives": fd_outputs["minus_objectives"],
            "mode_difference_counts": fd_outputs[
                "mode_difference_counts"
            ],
        }
    )
    results["crn_fd"] = crn_fd
    fd_finite = crn_fd["finite"]
    fd_nonzero = bool(fd_finite and crn_fd["l2_norm"] > NUMERICAL_ZERO_ATOL)
    mode_changed = bool(any(crn_fd["mode_difference_counts"]))
    results["gates"].update(
        {
            "crn_fd_finite": fd_finite,
            "crn_fd_nonzero": fd_nonzero,
            "hard_mode_history_changed": mode_changed,
        }
    )
    passed = bool(
        direct_ad["numerical_zero"]
        and fd_finite
        and fd_nonzero
        and mode_changed
    )
    if not fd_finite:
        reason = "CRN-FD contains a non-finite component."
    elif not fd_nonzero:
        reason = "CRN-FD is numerical-zero at the frozen tape bank."
    elif not mode_changed:
        reason = "FD perturbations did not change any hard mode history."
    else:
        reason = (
            "Raw Direct AD is zero while finite non-zero same-tape CRN-FD "
            "changes hard mode histories."
        )
    return _finish(results, "PASS" if passed else "FAIL", reason)


if __name__ == "__main__":
    raise SystemExit(main())
