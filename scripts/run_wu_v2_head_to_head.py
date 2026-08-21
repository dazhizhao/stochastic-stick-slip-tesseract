"""Compare frozen Wu-V2 Markov policies with the W1 deterministic baseline."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from stochastic_stick_slip.wu2019_reproduction import (
    LOCAL_FRF_RATIOS as W1_LOCAL_FRF_RATIOS,
)
from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    REFERENCE_PRELOAD,
    SYSTEM,
    diagnostic_steady_state_metrics,
    simulate_preload_bank,
    single_tone_forcing,
)
from stochastic_stick_slip.wu_v2_markov import (
    CONDITION_LABELS,
    MARKOV_BASE_SEED,
    NUM_STEPS,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    deterministic_binary_preload,
    evaluate_markov_bank,
    markov_uniform_bank,
    policy_polar_coordinates,
    transition_probabilities,
)


W1_PATH = ROOT / "outputs/wu2019_reproduction/scorecard.json"
GATE_0_PATH = ROOT / "outputs/wu_v2_gate0_final/results.json"
GATE_A_PATH = ROOT / "outputs/wu_v2_gate_a/results.json"
GATE_BC_PATH = ROOT / "outputs/wu_v2_gate_bc/results.json"
LANDSCAPE_PATH = ROOT / "outputs/wu_v2_landscape_diagnostic/results.json"
CANDIDATE_PATHS = {
    "stochastic_lr0p1": ROOT
    / "outputs/wu_v2_gate_bc_100_lr0p1/results.json",
    "stochastic_lr1p0": ROOT
    / "outputs/wu_v2_gate_bc_100_lr1/results.json",
}
OUTPUT_DIRECTORY = ROOT / "outputs/wu_v2_head_to_head"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.json"
MARKDOWN_PATH = OUTPUT_DIRECTORY / "head_to_head.md"
FRF_FIGURE_PATH = OUTPUT_DIRECTORY / "head_to_head_frf.png"
POLICY_FIGURE_PATH = OUTPUT_DIRECTORY / "learned_switching_policy.png"

CONFIRMATION_STREAMS = (5, 6, 7, 8)
NUM_REALIZATIONS_PER_CONDITION = 8
REALIZATIONS_PER_BANK = len(CONDITION_LABELS) * NUM_REALIZATIONS_PER_CONDITION
TOTAL_CONFIRMATION_REALIZATIONS = len(CONFIRMATION_STREAMS) * REALIZATIONS_PER_BANK
EXPECTED_OMEGA_RATIO = 1.19
EXPECTED_WU_AMPLITUDE_RATIO = 0.25
EXPECTED_WU_AMPLITUDE = 0.01
EXPECTED_WU_PHASE = 4.516039439535327
EXPECTED_PASSIVE_PEAK = 0.18748720511761083
EXPECTED_WU_PEAK = 0.1495599768466055
EXPECTED_CANDIDATES = {
    "stochastic_lr0p1": {
        "learning_rate": 0.1,
        "q": np.asarray([-3.514982271973227, -0.5337623522652377]),
    },
    "stochastic_lr1p0": {
        "learning_rate": 1.0,
        "q": np.asarray([-10.665739565561044, -6.033414703985564]),
    },
}
REFERENCE_RTOL = 1e-12
REFERENCE_ATOL = 1e-14

FRAME_COLOR = "#20242A"
PASSIVE_COLOR = "#777D84"
WU_COLOR = "#315F55"
BINARY_COLOR = "#376A8B"
LR01_COLOR = "#7D5687"
LR10_COLOR = "#B36A4C"
METHOD_COLORS = {
    "passive": PASSIVE_COLOR,
    "wu_continuous_2omega": WU_COLOR,
    "binary_deterministic_2omega": BINARY_COLOR,
    "stochastic_lr0p1": LR01_COLOR,
    "stochastic_lr1p0": LR10_COLOR,
}
METHOD_LABELS = {
    "passive": "Passive",
    "wu_continuous_2omega": "Wu-style continuous 2ω",
    "binary_deterministic_2omega": "Deterministic binary 2ω",
    "stochastic_lr0p1": "Stochastic, lr=0.1",
    "stochastic_lr1p0": "Stochastic, lr=1.0",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _allclose(left, right) -> bool:
    return bool(
        np.allclose(left, right, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    )


def _validate_candidate(name: str, result: dict, omega_r: float) -> np.ndarray:
    expected = EXPECTED_CANDIDATES[name]
    configuration = result["configuration"]
    optimization = result["optimization"]
    learned = result["learned_policy"]
    final_history = optimization["history"][-1]
    q = np.asarray(final_history["q"], dtype=np.float64)
    frozen = (
        configuration["num_periods"] == DIAGNOSTIC_NUM_PERIODS
        and configuration["steps_per_period"] == 100
        and configuration["streams"]["training"] == 2
        and configuration["streams"]["fixed_evaluation"] == 3
        and configuration["markov_base_seed"] == MARKOV_BASE_SEED
        and _allclose(configuration["omega_r"], omega_r)
        and _allclose(configuration["damping"], DAMPING)
        and _allclose(configuration["forcing_amplitude"], FORCING_AMPLITUDE)
        and _allclose(configuration["preload_low"], PRELOAD_LOW)
        and _allclose(configuration["preload_high"], PRELOAD_HIGH)
        and optimization["optimizer"]["num_updates"] == 100
        and optimization["optimizer"]["selected_iteration"] == 100
        and _allclose(
            optimization["optimizer"]["learning_rate"],
            expected["learning_rate"],
        )
        and final_history["iteration"] == 100
        and final_history["training_bank_iteration"] == 99
        and learned["final_iteration"] == 100
        and _allclose(q, learned["final_q"])
        and _allclose(q, expected["q"])
    )
    if not frozen:
        raise RuntimeError(f"Frozen q100 candidate mismatch: {name}")
    return q


def load_frozen_inputs() -> dict:
    w1 = _read_json(W1_PATH)
    gate_0 = _read_json(GATE_0_PATH)
    gate_a = _read_json(GATE_A_PATH)
    gate_bc = _read_json(GATE_BC_PATH)
    landscape = _read_json(LANDSCAPE_PATH)

    configuration = w1["configuration"]
    omega_r = float(configuration["omega_r"])
    ratios = np.asarray(configuration["local_frf_ratios"], dtype=np.float64)
    search = w1["harmonic_search"]["2"]
    passive = w1["local_frf"]["methods"]["passive"]
    wu_two = w1["local_frf"]["methods"]["2omega"]
    frozen = (
        w1["interpretation"]["category"] == "Partial reproduction"
        and gate_0["gate_0"]["result"] == "PASS"
        and gate_a["gate_a"]["result"] == "PASS"
        and gate_bc["gate_b"]["result"] == "PASS"
        and landscape["configuration"]["landscape_stream"] == 3
        and landscape["configuration"]["confirmation_stream"] == 4
        and _allclose(configuration["preload_A0"], REFERENCE_PRELOAD)
        and _allclose(configuration["omega_r_ratio"], EXPECTED_OMEGA_RATIO)
        and _allclose(ratios, W1_LOCAL_FRF_RATIOS)
        and _allclose(search["best_amplitude_ratio"], EXPECTED_WU_AMPLITUDE_RATIO)
        and _allclose(search["best_amplitude"], EXPECTED_WU_AMPLITUDE)
        and _allclose(search["best_phase_rad"], EXPECTED_WU_PHASE)
        and _allclose(passive["peak_amplitude"], EXPECTED_PASSIVE_PEAK)
        and _allclose(wu_two["peak_amplitude"], EXPECTED_WU_PEAK)
        and not passive["peak_at_boundary"]
        and not wu_two["peak_at_boundary"]
    )
    if not frozen:
        raise RuntimeError("W1 or Wu-V2 frozen references do not match W2")

    candidates = {}
    for name, path in CANDIDATE_PATHS.items():
        result = _read_json(path)
        q = _validate_candidate(name, result, omega_r)
        magnitude, phase = policy_polar_coordinates(q)
        candidates[name] = {
            "learning_rate": EXPECTED_CANDIDATES[name]["learning_rate"],
            "q": q,
            "magnitude": magnitude,
            "coefficient_phase": phase,
            "source_iteration": 100,
            "source_training_bank_iteration": 99,
        }

    return {
        "w1": w1,
        "omega_r": omega_r,
        "omega_r_ratio": float(configuration["omega_r_ratio"]),
        "frequency_ratios": ratios,
        "wu_amplitude": float(search["best_amplitude"]),
        "wu_amplitude_ratio": float(search["best_amplitude_ratio"]),
        "wu_phase": float(search["best_phase_rad"]),
        "passive": passive,
        "wu_two": wu_two,
        "candidates": candidates,
    }


def deterministic_peak_summary(
    ratios: np.ndarray,
    amplitudes: np.ndarray,
    steady_errors: np.ndarray | None = None,
) -> dict:
    ratios = np.asarray(ratios, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if ratios.ndim != 1 or amplitudes.shape != ratios.shape:
        raise ValueError("deterministic FRF arrays must have matching 1D shapes")
    if not np.all(np.isfinite(amplitudes)):
        raise FloatingPointError("deterministic FRF amplitude is non-finite")
    peak_index = int(np.argmax(amplitudes))
    boundary = peak_index in (0, len(ratios) - 1)
    result = {
        "steady_amplitudes": amplitudes.tolist(),
        "peak_index": peak_index,
        "peak_ratio": float(ratios[peak_index]),
        "peak_amplitude": float(amplitudes[peak_index]),
        "peak_at_boundary": boundary,
        "range_status": "range_insufficient" if boundary else "interior",
    }
    if steady_errors is not None:
        errors = np.asarray(steady_errors, dtype=np.float64)
        if errors.shape != ratios.shape or not np.all(np.isfinite(errors)):
            raise FloatingPointError("deterministic steady error is invalid")
        result["steady_errors"] = errors.tolist()
        result["peak_steady_error"] = float(errors[peak_index])
    return result


def stochastic_frf_summary(
    ratios: np.ndarray,
    objectives: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Summarize [frequency, bank, realization] amplitudes without raw output."""
    ratios = np.asarray(ratios, dtype=np.float64)
    values = np.asarray(objectives, dtype=np.float64)
    expected_shape = (
        len(ratios),
        len(CONFIRMATION_STREAMS),
        REALIZATIONS_PER_BANK,
    )
    if values.shape != expected_shape:
        raise ValueError(f"stochastic FRF values must have shape {expected_shape}")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("stochastic FRF amplitude is non-finite")
    flattened = values.reshape((len(ratios), -1))
    aggregate_mean = np.mean(flattened, axis=1)
    population_std = np.std(flattened, axis=1, ddof=0)
    realization_min = np.min(flattened, axis=1)
    realization_max = np.max(flattened, axis=1)
    bank_means = np.mean(values, axis=2).T
    peak_index = int(np.argmax(aggregate_mean))
    peak_boundary = peak_index in (0, len(ratios) - 1)
    bank_peaks = []
    for bank_index, stream_id in enumerate(CONFIRMATION_STREAMS):
        index = int(np.argmax(bank_means[bank_index]))
        boundary = index in (0, len(ratios) - 1)
        bank_peaks.append(
            {
                "stream_id": stream_id,
                "peak_index": index,
                "peak_ratio": float(ratios[index]),
                "peak_amplitude": float(bank_means[bank_index, index]),
                "peak_at_boundary": boundary,
                "range_status": (
                    "range_insufficient" if boundary else "interior"
                ),
            }
        )
    return (
        {
            "aggregate_mean_amplitudes": aggregate_mean.tolist(),
            "population_std_amplitudes": population_std.tolist(),
            "realization_min_amplitudes": realization_min.tolist(),
            "realization_max_amplitudes": realization_max.tolist(),
            "bank_mean_amplitudes": bank_means.tolist(),
            "aggregate_peak": {
                "peak_index": peak_index,
                "peak_ratio": float(ratios[peak_index]),
                "peak_amplitude": float(aggregate_mean[peak_index]),
                "peak_at_boundary": peak_boundary,
                "range_status": (
                    "range_insufficient" if peak_boundary else "interior"
                ),
            },
            "bank_peaks": bank_peaks,
        },
        flattened,
    )


def expected_high_probability(
    probability_low_to_high: np.ndarray,
    probability_high_to_low: np.ndarray,
) -> np.ndarray:
    probability_lh = np.asarray(probability_low_to_high, dtype=np.float64)
    probability_hl = np.asarray(probability_high_to_low, dtype=np.float64)
    if probability_lh.shape != probability_hl.shape or probability_lh.ndim != 1:
        raise ValueError("transition probabilities must be matching 1D arrays")
    high = 0.5
    history = np.empty_like(probability_lh)
    for index, (p_lh, p_hl) in enumerate(
        zip(probability_lh, probability_hl, strict=True)
    ):
        high = high * (1.0 - p_hl) + (1.0 - high) * p_lh
        history[index] = high
    return history


def _binary_local_frf(frozen: dict) -> tuple[dict, float]:
    started = time.perf_counter()
    amplitudes = []
    steady_errors = []
    for index, ratio in enumerate(frozen["frequency_ratios"]):
        omega = float(ratio * frozen["omega_r"])
        preload = deterministic_binary_preload(omega, frozen["wu_phase"])
        displacement = np.asarray(simulate_preload_bank(omega, preload)[0])
        objective, steady_error, _ = diagnostic_steady_state_metrics(displacement)
        values = np.concatenate((objective, steady_error))
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"binary result is non-finite at {ratio=}")
        amplitudes.append(float(objective[0]))
        steady_errors.append(float(steady_error[0]))
        print(f"binary_frf={index + 1:02d}/{len(frozen['frequency_ratios'])}")
    return (
        deterministic_peak_summary(
            frozen["frequency_ratios"], amplitudes, steady_errors
        ),
        time.perf_counter() - started,
    )


def _confirmation_banks() -> dict[int, np.ndarray]:
    banks = {
        stream_id: markov_uniform_bank(
            NUM_REALIZATIONS_PER_CONDITION, stream_id=stream_id, iteration=0
        )
        for stream_id in CONFIRMATION_STREAMS
    }
    for stream_id, bank in banks.items():
        if bank.shape != (8, 8, NUM_STEPS + 1, 2):
            raise AssertionError(f"confirmation bank shape mismatch: {stream_id}")
    for left_index, left in enumerate(CONFIRMATION_STREAMS):
        for right in CONFIRMATION_STREAMS[left_index + 1 :]:
            if np.array_equal(banks[left], banks[right]):
                raise AssertionError("confirmation banks are not independent")
    return banks


def _stochastic_local_frf(
    frozen: dict,
    confirmation_banks: dict[int, np.ndarray],
) -> tuple[dict, dict, dict, float]:
    started = time.perf_counter()
    summaries = {}
    diagnostics = {}
    realization_diagnostics = {}
    nominal_index_values = np.flatnonzero(
        np.isclose(frozen["frequency_ratios"], 1.0)
    )
    if nominal_index_values.size != 1:
        raise AssertionError("W1 FRF grid must contain one nominal frequency")
    nominal_index = int(nominal_index_values[0])

    for candidate_index, (name, candidate) in enumerate(
        frozen["candidates"].items()
    ):
        q = candidate["q"].copy()
        q_before = q.copy()
        values = np.empty(
            (
                len(frozen["frequency_ratios"]),
                len(CONFIRMATION_STREAMS),
                REALIZATIONS_PER_BANK,
            ),
            dtype=np.float64,
        )
        nominal_occupancy = []
        nominal_transitions = []
        representative_modes = None
        nominal_probabilities = None

        for frequency_index, ratio in enumerate(frozen["frequency_ratios"]):
            omega = float(ratio * frozen["omega_r"])
            time_step, forcing = single_tone_forcing(
                FORCING_AMPLITUDE, omega, DIAGNOSTIC_NUM_PERIODS
            )
            times = time_step * np.arange(1, NUM_STEPS + 1, dtype=np.float64)
            for bank_index, stream_id in enumerate(CONFIRMATION_STREAMS):
                result = evaluate_markov_bank(
                    q,
                    forcing,
                    confirmation_banks[stream_id],
                    times,
                    omega,
                    time_step,
                )
                objectives = np.asarray(result["trajectory_objectives"]).reshape(-1)
                if objectives.shape != (REALIZATIONS_PER_BANK,):
                    raise AssertionError("confirmation objective shape mismatch")
                values[frequency_index, bank_index] = objectives
                if frequency_index == nominal_index:
                    nominal_occupancy.append(
                        np.asarray(result["high_mode_fraction"]).reshape((-1, 2))
                    )
                    nominal_transitions.append(
                        np.asarray(result["transition_counts"]).reshape((-1, 2))
                    )
                    if stream_id == CONFIRMATION_STREAMS[0]:
                        representative_modes = np.asarray(result["modes"])[
                            0, 0, :, 0
                        ].astype(bool)
            if frequency_index == nominal_index:
                p_lh, p_hl = transition_probabilities(
                    q, times, omega, time_step
                )
                nominal_probabilities = (
                    np.asarray(p_lh, dtype=np.float64),
                    np.asarray(p_hl, dtype=np.float64),
                )
            print(
                f"stochastic_candidate={candidate_index + 1}/2 "
                f"frequency={frequency_index + 1:02d}/"
                f"{len(frozen['frequency_ratios'])}"
            )

        if not np.array_equal(q, q_before):
            raise AssertionError(f"candidate q changed during evaluation: {name}")
        summary, flattened = stochastic_frf_summary(
            frozen["frequency_ratios"], values
        )
        peak_index = summary["aggregate_peak"]["peak_index"]
        peak_realizations = flattened[peak_index]
        below = peak_realizations < frozen["wu_two"]["peak_amplitude"]
        realization_diagnostics[name] = {
            "frequency_index": peak_index,
            "frequency_ratio": float(frozen["frequency_ratios"][peak_index]),
            "wu_local_peak_threshold": float(
                frozen["wu_two"]["peak_amplitude"]
            ),
            "count_below_wu_local_peak": int(np.count_nonzero(below)),
            "fraction_below_wu_local_peak": float(np.mean(below)),
            "num_realizations": TOTAL_CONFIRMATION_REALIZATIONS,
            "diagnostic_only": True,
        }
        summaries[name] = summary

        occupancy = np.concatenate(nominal_occupancy, axis=0)
        transitions = np.concatenate(nominal_transitions, axis=0)
        p_lh, p_hl = nominal_probabilities
        diagnostics[name] = {
            "q": q.tolist(),
            "magnitude": candidate["magnitude"],
            "coefficient_phase": candidate["coefficient_phase"],
            "phase_convention": "h=R*cos(2*omega*t-coefficient_phase)",
            "mean_high_occupancy_per_contact": np.mean(occupancy, axis=0).tolist(),
            "mean_transitions_per_trajectory_contact": np.mean(
                transitions, axis=0
            ).tolist(),
            "transition_probability_low_to_high_min_max": [
                float(np.min(p_lh)),
                float(np.max(p_lh)),
            ],
            "transition_probability_high_to_low_min_max": [
                float(np.min(p_hl)),
                float(np.max(p_hl)),
            ],
            "expected_high_probability": expected_high_probability(
                p_lh, p_hl
            ),
            "representative_modes": representative_modes,
        }
    return (
        summaries,
        diagnostics,
        realization_diagnostics,
        time.perf_counter() - started,
    )


def _nominal_summary(local_frf: dict, nominal_index: int) -> dict:
    methods = {}
    for name in (
        "passive",
        "wu_continuous_2omega",
        "binary_deterministic_2omega",
    ):
        method = local_frf["methods"][name]
        methods[name] = {
            "amplitude": float(method["steady_amplitudes"][nominal_index])
        }
        if "steady_errors" in method:
            methods[name]["steady_error"] = float(
                method["steady_errors"][nominal_index]
            )
    for name in ("stochastic_lr0p1", "stochastic_lr1p0"):
        method = local_frf["methods"][name]
        bank_values = np.asarray(method["bank_mean_amplitudes"])[
            :, nominal_index
        ]
        methods[name] = {
            "overall_mean_amplitude": float(
                method["aggregate_mean_amplitudes"][nominal_index]
            ),
            "bank_mean_amplitudes": bank_values.tolist(),
            "population_std_amplitude": float(
                method["population_std_amplitudes"][nominal_index]
            ),
            "realization_range": [
                float(method["realization_min_amplitudes"][nominal_index]),
                float(method["realization_max_amplitudes"][nominal_index]),
            ],
        }
    return {"frequency_ratio": 1.0, "methods": methods}


def _performance_comparisons(local_frf: dict) -> dict:
    passive_peak = local_frf["methods"]["passive"]["peak_amplitude"]
    wu_peak = local_frf["methods"]["wu_continuous_2omega"]["peak_amplitude"]
    binary_peak = local_frf["methods"]["binary_deterministic_2omega"][
        "peak_amplitude"
    ]
    table = {}
    for name, method in local_frf["methods"].items():
        peak = (
            method["aggregate_peak"]
            if name.startswith("stochastic_")
            else method
        )
        amplitude = float(peak["peak_amplitude"])
        table[name] = {
            "local_peak_amplitude": amplitude,
            "local_peak_ratio": float(peak["peak_ratio"]),
            "range_status": peak["range_status"],
            "reduction_vs_passive_percent": float(
                100.0 * (passive_peak - amplitude) / passive_peak
            ),
            "improvement_vs_wu_percent": float(
                100.0 * (wu_peak - amplitude) / wu_peak
            ),
        }
        if name.startswith("stochastic_"):
            table[name]["improvement_vs_binary_percent"] = float(
                100.0 * (binary_peak - amplitude) / binary_peak
            )
    return {
        "definition": "positive improvement means the named method is lower",
        "methods": table,
    }


def _conclusion(local_frf: dict, comparisons: dict) -> dict:
    candidate_names = ("stochastic_lr0p1", "stochastic_lr1p0")
    candidate_peaks = {
        name: local_frf["methods"][name]["aggregate_peak"]["peak_amplitude"]
        for name in candidate_names
    }
    selected = min(candidate_names, key=lambda name: candidate_peaks[name])
    wu_peak = comparisons["methods"]["wu_continuous_2omega"][
        "local_peak_amplitude"
    ]
    binary_peak = comparisons["methods"]["binary_deterministic_2omega"][
        "local_peak_amplitude"
    ]
    outcomes = {}
    for name in candidate_names:
        peak = local_frf["methods"][name]["aggregate_peak"]
        amplitude = peak["peak_amplitude"]
        if peak["peak_at_boundary"]:
            category = "range_insufficient"
            explanation = (
                "The sampled-window maximum is reported, but the true local "
                "resonance peak is not established."
            )
        elif amplitude < wu_peak and amplitude < binary_peak:
            category = "stochastic_switching_adds_value"
            explanation = (
                "The stochastic policy is below both deterministic comparators."
            )
        elif amplitude < wu_peak:
            category = "binary_quantization_dominates"
            explanation = (
                "The stochastic policy is below Wu-style continuous control "
                "but not below deterministic binary control."
            )
        else:
            category = "does_not_outperform_wu_local_peak"
            explanation = (
                "The fixed-frequency advantage does not extend to the local-peak "
                "objective."
            )
        outcomes[name] = {"category": category, "explanation": explanation}

    selected_peak = local_frf["methods"][selected]["aggregate_peak"]
    if not selected_peak["peak_at_boundary"] and selected_peak["peak_amplitude"] < wu_peak:
        statement = (
            "The optimized stochastic Markov policy outperforms the Wu-style "
            "deterministic 2ω sinusoidal control on the same JAX-FEM benchmark."
        )
    elif selected_peak["peak_at_boundary"]:
        statement = (
            "The stochastic sampled-window result is range-insufficient, so a "
            "true local-peak outperformance claim is withheld."
        )
    else:
        statement = (
            "Neither frozen stochastic candidate outperforms the Wu-style "
            "deterministic 2ω local peak on this JAX-FEM benchmark."
        )
    return {
        "not_a_performance_gate": True,
        "visualization_candidate": selected,
        "visualization_selection_rule": (
            "smaller aggregate local peak; lr=0.1 wins an exact tie"
        ),
        "candidate_outcomes": outcomes,
        "statement": statement,
        "claim_boundary": (
            "This compares implementations on the same JAX-FEM benchmark; it "
            "does not claim to outperform Wu et al. 2019."
        ),
    }


def _json_ready_diagnostics(diagnostics: dict) -> dict:
    cleaned = {}
    for name, values in diagnostics.items():
        cleaned[name] = {
            key: value
            for key, value in values.items()
            if key not in ("expected_high_probability", "representative_modes")
        }
    return cleaned


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _style_axis(axis) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME_COLOR)
        spine.set_linewidth(1.1)
    axis.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=False,
        right=False,
        width=0.9,
        colors=FRAME_COLOR,
    )


def _plot_head_to_head(results: dict) -> None:
    _configure_plotting()
    ratios = np.asarray(results["local_frf"]["frequency_ratios"])
    methods = results["local_frf"]["methods"]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.3),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    axis = axes[0]
    for name in (
        "passive",
        "wu_continuous_2omega",
        "binary_deterministic_2omega",
        "stochastic_lr0p1",
        "stochastic_lr1p0",
    ):
        method = methods[name]
        if name.startswith("stochastic_"):
            curve = np.asarray(method["aggregate_mean_amplitudes"])
            bank_curves = np.asarray(method["bank_mean_amplitudes"])
            axis.fill_between(
                ratios,
                np.min(bank_curves, axis=0),
                np.max(bank_curves, axis=0),
                color=METHOD_COLORS[name],
                alpha=0.10,
                linewidth=0.0,
            )
        else:
            curve = np.asarray(method["steady_amplitudes"])
        axis.plot(
            ratios,
            curve,
            color=METHOD_COLORS[name],
            linewidth=2.0 if name != "passive" else 1.7,
            marker="o",
            markersize=3.0,
            label=METHOD_LABELS[name],
        )
    axis.set_xlabel(r"Frequency ratio, $\omega/\omega_r$")
    axis.set_ylabel("Steady amplitude")
    axis.legend(loc="best", handlelength=2.3)
    axis.text(-0.12, 1.03, "a", transform=axis.transAxes, fontweight="bold")
    _style_axis(axis)

    axis = axes[1]
    active_names = (
        "wu_continuous_2omega",
        "binary_deterministic_2omega",
        "stochastic_lr0p1",
        "stochastic_lr1p0",
    )
    values = [
        results["comparisons"]["methods"][name][
            "reduction_vs_passive_percent"
        ]
        for name in active_names
    ]
    labels = ["Wu cont.\n2ω", "Binary\n2ω", "Stoch.\nlr=0.1", "Stoch.\nlr=1.0"]
    bars = axis.bar(
        np.arange(len(active_names)),
        values,
        color=[METHOD_COLORS[name] for name in active_names],
        width=0.68,
    )
    axis.axhline(0.0, color=FRAME_COLOR, linewidth=0.8)
    axis.set_xticks(np.arange(len(active_names)), labels)
    axis.set_ylabel("Peak reduction vs passive (%)")
    padding = max(0.3, 0.025 * max(abs(np.asarray(values))))
    for bar, value in zip(bars, values, strict=True):
        vertical = "bottom" if value >= 0.0 else "top"
        offset = padding if value >= 0.0 else -padding
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f"{value:.2f}%",
            ha="center",
            va=vertical,
            fontsize=7.5,
            color=FRAME_COLOR,
        )
    axis.text(-0.16, 1.03, "b", transform=axis.transAxes, fontweight="bold")
    _style_axis(axis)
    figure.tight_layout(w_pad=2.6)
    figure.savefig(FRF_FIGURE_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _plot_policy(results: dict, diagnostics: dict) -> None:
    _configure_plotting()
    selected = results["conclusion"]["visualization_candidate"]
    values = diagnostics[selected]
    final_slice = slice(-100, None)
    phase_fraction = np.arange(1, 101, dtype=np.float64) / 100.0
    forcing_phase = 2.0 * np.pi * phase_fraction
    wu = REFERENCE_PRELOAD + results["frozen_references"][
        "wu_2omega_amplitude"
    ] * np.sin(
        2.0 * forcing_phase + results["frozen_references"]["wu_2omega_phase"]
    )
    binary = np.where(
        np.sin(
            2.0 * forcing_phase
            + results["frozen_references"]["wu_2omega_phase"]
        )
        >= 0.0,
        PRELOAD_HIGH,
        PRELOAD_LOW,
    )
    expected = values["expected_high_probability"][final_slice]
    representative = values["representative_modes"][final_slice].astype(float)

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axis = axes[0]
    axis.plot(
        phase_fraction,
        wu,
        color=WU_COLOR,
        linewidth=2.0,
        label="Wu-style continuous 2ω",
    )
    axis.step(
        phase_fraction,
        binary,
        where="post",
        color=BINARY_COLOR,
        linewidth=1.8,
        label="Deterministic binary 2ω",
    )
    axis.set_xlabel("Forcing-cycle phase")
    axis.set_ylabel("Preload")
    axis.set_xlim(0.0, 1.0)
    axis.legend(loc="best")
    axis.text(-0.12, 1.03, "a", transform=axis.transAxes, fontweight="bold")
    _style_axis(axis)

    axis = axes[1]
    axis.plot(
        phase_fraction,
        expected,
        color=METHOD_COLORS[selected],
        linewidth=2.1,
        label="Expected HIGH probability",
    )
    axis.step(
        phase_fraction,
        representative,
        where="post",
        color=FRAME_COLOR,
        linewidth=1.2,
        alpha=0.72,
        label="Fixed representative path",
    )
    axis.set_xlabel("Forcing-cycle phase")
    axis.set_ylabel("HIGH state")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.04, 1.04)
    axis.legend(loc="best")
    axis.text(-0.12, 1.03, "b", transform=axis.transAxes, fontweight="bold")
    _style_axis(axis)
    figure.tight_layout(w_pad=2.6)
    figure.savefig(
        POLICY_FIGURE_PATH, dpi=600, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)


def _write_markdown(results: dict) -> None:
    nominal = results["nominal"]["methods"]
    comparisons = results["comparisons"]["methods"]
    lines = [
        "# Wu-V2 stochastic Markov vs Wu-style deterministic 2ω",
        "",
        "All stochastic results use four new independent confirmation banks "
        "(64 Markov realizations per bank, 256 total).",
        "",
        "## Head-to-head scorecard",
        "",
        "| Method | Nominal response | Local peak | vs Passive | vs Wu 2ω |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "passive",
        "wu_continuous_2omega",
        "binary_deterministic_2omega",
        "stochastic_lr0p1",
        "stochastic_lr1p0",
    ):
        nominal_value = (
            nominal[name]["overall_mean_amplitude"]
            if name.startswith("stochastic_")
            else nominal[name]["amplitude"]
        )
        comparison = comparisons[name]
        versus_wu = (
            "—"
            if name == "wu_continuous_2omega"
            else f"{comparison['improvement_vs_wu_percent']:.4f}%"
        )
        lines.append(
            f"| {METHOD_LABELS[name]} | {nominal_value:.9f} | "
            f"{comparison['local_peak_amplitude']:.9f} | "
            f"{comparison['reduction_vs_passive_percent']:.4f}% | "
            f"{versus_wu} |"
        )

    lines.extend(
        [
            "",
            "## Independent-bank sampled peaks",
            "",
            "| Policy | Aggregate | Stream 5 | Stream 6 | Stream 7 | Stream 8 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("stochastic_lr0p1", "stochastic_lr1p0"):
        method = results["local_frf"]["methods"][name]
        bank_values = [peak["peak_amplitude"] for peak in method["bank_peaks"]]
        lines.append(
            f"| {METHOD_LABELS[name]} | "
            f"{method['aggregate_peak']['peak_amplitude']:.9f} | "
            + " | ".join(f"{value:.9f}" for value in bank_values)
            + " |"
        )

    lines.extend(["", "## Markov policy diagnostics", ""])
    for name in ("stochastic_lr0p1", "stochastic_lr1p0"):
        diagnostic = results["markov_diagnostics"][name]
        realization = results["realization_diagnostic"][name]
        lines.extend(
            [
                f"### {METHOD_LABELS[name]}",
                "",
                f"- q: `{diagnostic['q']}`",
                f"- magnitude: `{diagnostic['magnitude']:.9f}`",
                f"- coefficient phase: `{diagnostic['coefficient_phase']:.9f} rad`",
                "- mean HIGH occupancy/contact: "
                f"`{diagnostic['mean_high_occupancy_per_contact']}`",
                "- mean transitions/trajectory/contact: "
                f"`{diagnostic['mean_transitions_per_trajectory_contact']}`",
                "- realizations below the Wu local peak at the stochastic "
                f"aggregate peak: `{realization['count_below_wu_local_peak']}/"
                f"{realization['num_realizations']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "",
            results["conclusion"]["statement"],
            "",
            results["conclusion"]["claim_boundary"],
            "",
        ]
    )
    MARKDOWN_PATH.write_text("\n".join(lines))


def main() -> None:
    total_started = time.perf_counter()
    frozen = load_frozen_inputs()
    print("frozen_inputs=validated")
    confirmation_banks = _confirmation_banks()
    print("confirmation_streams=5,6,7,8")

    binary, binary_seconds = _binary_local_frf(frozen)
    stochastic, diagnostics, realization_diagnostics, stochastic_seconds = (
        _stochastic_local_frf(frozen, confirmation_banks)
    )
    ratios = frozen["frequency_ratios"]
    passive = deterministic_peak_summary(
        ratios,
        frozen["passive"]["steady_amplitudes"],
        frozen["passive"]["steady_errors"],
    )
    wu_two = deterministic_peak_summary(
        ratios,
        frozen["wu_two"]["steady_amplitudes"],
        frozen["wu_two"]["steady_errors"],
    )
    local_frf = {
        "frequency_ratios": ratios.tolist(),
        "peak_definition": "maximum over the 21 sampled frequency points",
        "stochastic_peak_definition": "max_frequency mean_over_256_realizations",
        "mean_of_per_realization_maxima_used": False,
        "methods": {
            "passive": passive,
            "wu_continuous_2omega": wu_two,
            "binary_deterministic_2omega": binary,
            **stochastic,
        },
    }
    nominal_index = int(np.flatnonzero(np.isclose(ratios, 1.0))[0])
    comparisons = _performance_comparisons(local_frf)
    conclusion = _conclusion(local_frf, comparisons)
    runtime = {
        "binary_local_frf_seconds": binary_seconds,
        "stochastic_local_frf_seconds": stochastic_seconds,
        "scientific_evaluation_seconds": time.perf_counter() - total_started,
        "binary_forward_calls": len(ratios),
        "stochastic_bank_forward_calls": (
            len(frozen["candidates"])
            * len(CONFIRMATION_STREAMS)
            * len(ratios)
        ),
        "training_or_gradient_calls": 0,
    }
    results = {
        "configuration": {
            "mesh": "32x4 QUAD4",
            "num_free_dofs": SYSTEM.num_free_dofs,
            "damping": DAMPING,
            "forcing_amplitude": FORCING_AMPLITUDE,
            "preload_reference": REFERENCE_PRELOAD,
            "preload_low": PRELOAD_LOW,
            "preload_high": PRELOAD_HIGH,
            "omega_r": frozen["omega_r"],
            "omega_r_ratio": frozen["omega_r_ratio"],
            "num_periods": DIAGNOSTIC_NUM_PERIODS,
            "steps_per_period": 100,
            "objective_cycles": [21, 22, 23, 24],
            "frequency_ratios": ratios.tolist(),
            "markov_base_seed": MARKOV_BASE_SEED,
            "confirmation_streams": list(CONFIRMATION_STREAMS),
            "realizations_per_bank": REALIZATIONS_PER_BANK,
            "total_confirmation_realizations": TOTAL_CONFIRMATION_REALIZATIONS,
            "all_realizations_share_nominal_forcing": True,
            "same_uniform_banks_reused_across_policies_and_frequencies": True,
        },
        "frozen_references": {
            "w1_category": frozen["w1"]["interpretation"]["category"],
            "passive_preload": REFERENCE_PRELOAD,
            "passive_local_peak": passive["peak_amplitude"],
            "wu_2omega_amplitude_ratio": frozen["wu_amplitude_ratio"],
            "wu_2omega_amplitude": frozen["wu_amplitude"],
            "wu_2omega_phase": frozen["wu_phase"],
            "wu_2omega_local_peak": wu_two["peak_amplitude"],
            "wu_2omega_local_peak_ratio": wu_two["peak_ratio"],
            "references_recomputed": False,
        },
        "candidates": {
            name: {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in candidate.items()
            }
            for name, candidate in frozen["candidates"].items()
        },
        "confirmation_banks": [
            {
                "stream_id": stream_id,
                "iteration": 0,
                "shape": [8, 8, NUM_STEPS + 1, 2],
                "num_markov_realizations": REALIZATIONS_PER_BANK,
                "used_for_training_or_candidate_selection": False,
            }
            for stream_id in CONFIRMATION_STREAMS
        ],
        "nominal": _nominal_summary(local_frf, nominal_index),
        "local_frf": local_frf,
        "realization_diagnostic": realization_diagnostics,
        "markov_diagnostics": _json_ready_diagnostics(diagnostics),
        "comparisons": comparisons,
        "conclusion": conclusion,
        "runtime": runtime,
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    _write_markdown(results)
    _plot_head_to_head(results)
    _plot_policy(results, diagnostics)
    runtime["total_runner_seconds"] = time.perf_counter() - total_started
    RESULTS_PATH.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    print("## W2 head-to-head")
    for name in (
        "passive",
        "wu_continuous_2omega",
        "binary_deterministic_2omega",
        "stochastic_lr0p1",
        "stochastic_lr1p0",
    ):
        entry = comparisons["methods"][name]
        print(
            f"{name}: local_peak={entry['local_peak_amplitude']:.16g} "
            f"ratio={entry['local_peak_ratio']:.3f} "
            f"vs_passive={entry['reduction_vs_passive_percent']:.9g}% "
            f"vs_wu={entry['improvement_vs_wu_percent']:.9g}%"
        )
    print(f"conclusion: {conclusion['statement']}")
    print(f"results: {RESULTS_PATH}")
    print(f"figure: {FRF_FIGURE_PATH}")
    print(f"policy_figure: {POLICY_FIGURE_PATH}")


if __name__ == "__main__":
    main()
