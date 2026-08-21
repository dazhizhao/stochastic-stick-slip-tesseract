"""Diagnose Wu-V2 binary authority and the fixed-bank Markov landscape."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np

from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    LOCAL_FRF_RATIOS,
    REFERENCE_PRELOAD,
    SYSTEM,
    diagnostic_steady_state_metrics,
    simulate_preload_bank,
    single_tone_forcing,
)
from stochastic_stick_slip.wu_v2_markov import (
    CONDITION_LABELS,
    LANDSCAPE_RADII,
    MARKOV_BASE_SEED,
    NUM_STEPS,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    deterministic_binary_preload,
    evaluate_markov_bank,
    landscape_polar_grid,
    markov_uniform_bank,
)


GATE_0_PATH = ROOT / "outputs/wu_v2_gate0_final/results.json"
GATE_A_PATH = ROOT / "outputs/wu_v2_gate_a/results.json"
GATE_BC_PATH = ROOT / "outputs/wu_v2_gate_bc/results.json"
OUTPUT_DIRECTORY = ROOT / "outputs/wu_v2_landscape_diagnostic"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.json"
FIGURE_PATH = OUTPUT_DIRECTORY / "landscape_diagnostic.png"

LANDSCAPE_STREAM = 3
CONFIRMATION_STREAM = 4
NUM_BANK_REALIZATIONS = 8
EXPECTED_OMEGA_RATIO = 1.19
EXPECTED_TWO_OMEGA_PHASE = 4.516039439535327
REFERENCE_RTOL = 1e-12
REFERENCE_ATOL = 1e-14

FRAME_COLOR = "#20242A"
PASSIVE_COLOR = "#737A83"
CONTINUOUS_COLOR = "#315F55"
BINARY_COLOR = "#376A8B"
STOCHASTIC_COLOR = "#7D5687"
RADIUS_COLORS = ("#8EA9BD", "#527E99", "#214E68")


def _load_frozen_inputs() -> tuple[float, float, dict]:
    gate_0 = json.loads(GATE_0_PATH.read_text())
    gate_a = json.loads(GATE_A_PATH.read_text())
    gate_bc = json.loads(GATE_BC_PATH.read_text())
    if gate_0["gate_0"]["result"] != "PASS":
        raise RuntimeError("Final Wu-style Gate 0 is not PASS")
    if gate_a["gate_a"]["result"] != "PASS":
        raise RuntimeError("Wu-V2 Gate A is not PASS")
    if gate_bc["gate_b"]["result"] != "PASS":
        raise RuntimeError("Wu-V2 Gate B is not PASS")
    if gate_bc["gate_c"]["result"] != "FAIL":
        raise RuntimeError("Wu-V2 Gate C is not the frozen FAIL result")

    configuration = gate_bc["configuration"]
    gate_a_configuration = gate_a["configuration"]
    frozen_values_match = (
        configuration["num_periods"] == DIAGNOSTIC_NUM_PERIODS
        and configuration["steps_per_period"] == 100
        and configuration["fixed_evaluation_realizations"] == 64
        and configuration["streams"]["fixed_evaluation"] == LANDSCAPE_STREAM
        and np.isclose(configuration["damping"], DAMPING)
        and np.isclose(configuration["forcing_amplitude"], FORCING_AMPLITUDE)
        and np.isclose(configuration["preload_low"], PRELOAD_LOW)
        and np.isclose(configuration["preload_high"], PRELOAD_HIGH)
        and gate_a_configuration["markov_base_seed"] == MARKOV_BASE_SEED
    )
    if not frozen_values_match:
        raise RuntimeError("A2/A3 artifacts do not match the frozen diagnosis")

    omega = float(configuration["omega_r"])
    omega_ratio = float(configuration["omega_r_ratio"])
    passive = float(gate_0["passive"]["steady_amplitude"])
    continuous = float(
        gate_0["phase_sweep"]["two_omega"]["best_amplitude"]
    )
    continuous_phase = float(
        gate_0["phase_sweep"]["two_omega"]["best_phase"]
    )
    recorded_neutral = float(
        gate_bc["optimization"]["initial_fixed_evaluation_amplitude"]
    )
    references_match = (
        np.isclose(REFERENCE_PRELOAD, gate_0["passive"]["preload"])
        and np.isclose(omega_ratio, EXPECTED_OMEGA_RATIO)
        and np.isclose(omega, gate_a_configuration["omega_r"])
        and np.isclose(continuous_phase, EXPECTED_TWO_OMEGA_PHASE)
        and np.isclose(gate_bc["wu_references"]["passive"], passive)
        and np.isclose(
            gate_bc["wu_references"]["best_deterministic_2omega"],
            continuous,
        )
        and np.isclose(
            gate_bc["wu_references"]["best_deterministic_2omega_phase"],
            continuous_phase,
        )
    )
    if not references_match:
        raise RuntimeError("Frozen Wu references do not match Final Gate 0")

    references = {
        "passive_amplitude": passive,
        "continuous_2omega_amplitude": continuous,
        "continuous_2omega_phase": continuous_phase,
        "stochastic_gate_a_neutral_amplitude": float(
            gate_a["neutral"]["mean_amplitude"]
        ),
        "stochastic_a3_fixed_neutral_amplitude": recorded_neutral,
    }
    return omega, omega_ratio, references


def _binary_local_frf(omega: float, phase: float) -> tuple[dict, dict]:
    steady_amplitudes = []
    steady_errors = []
    cycle_amplitudes = []
    for ratio in LOCAL_FRF_RATIOS:
        current_omega = float(ratio * omega)
        preload = deterministic_binary_preload(current_omega, phase)
        outputs = simulate_preload_bank(current_omega, preload)
        output_arrays = [np.asarray(output) for output in outputs]
        if not all(np.all(np.isfinite(output)) for output in output_arrays):
            raise FloatingPointError(
                f"Non-finite binary mechanics output at ratio={ratio}"
            )
        objective, steady_error, cycles = diagnostic_steady_state_metrics(
            output_arrays[0]
        )
        metrics = np.concatenate((objective, steady_error, cycles.ravel()))
        if not np.all(np.isfinite(metrics)):
            raise FloatingPointError(
                f"Non-finite binary metric at ratio={ratio}"
            )
        steady_amplitudes.append(float(objective[0]))
        steady_errors.append(float(steady_error[0]))
        cycle_amplitudes.append(cycles[0].tolist())

    amplitudes = np.asarray(steady_amplitudes)
    errors = np.asarray(steady_errors)
    peak_index = int(np.argmax(amplitudes))
    nominal_indices = np.flatnonzero(np.isclose(LOCAL_FRF_RATIOS, 1.0))
    if nominal_indices.size != 1:
        raise AssertionError("Local FRF grid does not contain one nominal point")
    nominal_index = int(nominal_indices[0])
    peak_at_boundary = peak_index in (0, len(LOCAL_FRF_RATIOS) - 1)
    local_frf = {
        "frequency_ratios": LOCAL_FRF_RATIOS.tolist(),
        "omegas": (LOCAL_FRF_RATIOS * omega).tolist(),
        "steady_amplitudes": steady_amplitudes,
        "steady_errors": steady_errors,
        "peak_index": peak_index,
        "peak_ratio": float(LOCAL_FRF_RATIOS[peak_index]),
        "peak_omega": float(LOCAL_FRF_RATIOS[peak_index] * omega),
        "peak_amplitude": float(amplitudes[peak_index]),
        "peak_steady_error": float(errors[peak_index]),
        "peak_at_boundary": peak_at_boundary,
        "range_status": (
            "range_insufficient" if peak_at_boundary else "interior"
        ),
    }
    nominal = {
        "amplitude": float(amplitudes[nominal_index]),
        "steady_error": float(errors[nominal_index]),
        "cycle_amplitudes": cycle_amplitudes[nominal_index],
        "local_frf_index": nominal_index,
    }
    return nominal, local_frf


def _markov_point(
    q: np.ndarray,
    radius: float,
    phase: float,
    forcing: np.ndarray,
    uniforms: np.ndarray,
    times: np.ndarray,
    omega: float,
    time_step: float,
) -> dict:
    evaluation = evaluate_markov_bank(
        q, forcing, uniforms, times, omega, time_step
    )
    objectives = np.asarray(evaluation["trajectory_objectives"])
    occupancy = np.asarray(evaluation["high_mode_fraction"])
    transitions = np.asarray(evaluation["transition_counts"])
    if not all(
        np.all(np.isfinite(values))
        for values in (objectives, occupancy, transitions)
    ):
        raise FloatingPointError(f"Non-finite Markov result at q={q.tolist()}")
    neutral = bool(radius == 0.0)
    return {
        "q": q.tolist(),
        "magnitude": float(radius),
        "phase": None if neutral else float(phase),
        "phase_fraction": (
            None if neutral else float(phase / (2.0 * np.pi))
        ),
        "mean_amplitude": float(np.mean(objectives)),
        "population_std_amplitude": float(np.std(objectives, ddof=0)),
        "mean_high_occupancy_per_contact": np.mean(
            occupancy, axis=(0, 1)
        ).tolist(),
        "mean_transitions_per_trajectory_contact": np.mean(
            transitions, axis=(0, 1)
        ).tolist(),
    }


def _landscape(
    forcing: np.ndarray,
    times: np.ndarray,
    omega: float,
    time_step: float,
    expected_neutral: float,
) -> tuple[dict, np.ndarray]:
    uniforms = markov_uniform_bank(
        NUM_BANK_REALIZATIONS, LANDSCAPE_STREAM, 0
    )
    q_values, radii, phases = landscape_polar_grid()
    points = [
        _markov_point(
            q_values[0],
            radii[0],
            phases[0],
            forcing,
            uniforms,
            times,
            omega,
            time_step,
        )
    ]
    neutral_amplitude = points[0]["mean_amplitude"]
    if not np.isclose(
        neutral_amplitude,
        expected_neutral,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    ):
        raise AssertionError("Landscape neutral does not reproduce A3")
    for q, radius, phase in zip(
        q_values[1:], radii[1:], phases[1:], strict=True
    ):
        points.append(
            _markov_point(
                q,
                radius,
                phase,
                forcing,
                uniforms,
                times,
                omega,
                time_step,
            )
        )
    for point in points:
        point["relative_change_percent"] = float(
            100.0
            * (point["mean_amplitude"] - neutral_amplitude)
            / neutral_amplitude
        )
    amplitudes = np.asarray([point["mean_amplitude"] for point in points])
    best_index = int(np.argmin(amplitudes))
    best = dict(points[best_index])
    best["grid_index"] = best_index
    return {
        "stream_id": LANDSCAPE_STREAM,
        "iteration": 0,
        "num_markov_realizations": int(
            len(CONDITION_LABELS) * NUM_BANK_REALIZATIONS
        ),
        "neutral_amplitude": neutral_amplitude,
        "points": points,
        "best_point": best,
    }, uniforms


def _confirmation(
    best_q: np.ndarray,
    forcing: np.ndarray,
    times: np.ndarray,
    omega: float,
    time_step: float,
    landscape_uniforms: np.ndarray,
) -> dict:
    uniforms = markov_uniform_bank(
        NUM_BANK_REALIZATIONS, CONFIRMATION_STREAM, 0
    )
    if np.array_equal(uniforms, landscape_uniforms):
        raise AssertionError("Landscape and confirmation banks are identical")
    neutral = _markov_point(
        np.zeros(2),
        0.0,
        0.0,
        forcing,
        uniforms,
        times,
        omega,
        time_step,
    )
    magnitude = float(np.linalg.norm(best_q))
    phase = float(np.mod(np.arctan2(best_q[1], best_q[0]), 2.0 * np.pi))
    best = _markov_point(
        best_q,
        magnitude,
        phase,
        forcing,
        uniforms,
        times,
        omega,
        time_step,
    )
    relative_change = float(
        100.0
        * (best["mean_amplitude"] - neutral["mean_amplitude"])
        / neutral["mean_amplitude"]
    )
    return {
        "stream_id": CONFIRMATION_STREAM,
        "iteration": 0,
        "num_markov_realizations": int(
            len(CONDITION_LABELS) * NUM_BANK_REALIZATIONS
        ),
        "neutral": neutral,
        "best_q": best,
        "best_q_relative_change_percent": relative_change,
        "best_q_below_neutral": bool(
            best["mean_amplitude"] < neutral["mean_amplitude"]
        ),
    }


def _diagnosis(
    binary_amplitude: float,
    passive_amplitude: float,
    landscape: dict,
    confirmation: dict,
) -> dict:
    landscape_better = bool(
        landscape["best_point"]["mean_amplitude"]
        < landscape["neutral_amplitude"]
    )
    confirmation_better = bool(confirmation["best_q_below_neutral"])
    if binary_amplitude >= passive_amplitude:
        case = "C"
        attribution = "quantization"
        reason = (
            "The deterministic binary command did not beat passive, so the "
            "LOW/HIGH quantization is the primary authority bottleneck."
        )
    elif landscape_better and confirmation_better:
        case = "A"
        attribution = "stochastic_optimization"
        reason = (
            "Binary control retained authority and the sampled Markov policy "
            "beat neutral on both independent banks; A3 failure therefore "
            "points to stochastic optimization or step scale."
        )
    else:
        case = "B"
        attribution = "markov_rate_mapping"
        reason = (
            "Binary control retained authority, but the registered Markov "
            "grid did not provide an independently confirmed advantage; the "
            "current rate mapping lacks a robust useful region."
        )
    return {
        "case": case,
        "a3_gate_c_failure_attribution": attribution,
        "binary_below_passive": bool(binary_amplitude < passive_amplitude),
        "landscape_best_below_neutral": landscape_better,
        "confirmation_best_below_neutral": confirmation_better,
        "reason": reason,
    }


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
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
        direction="in",
        top=False,
        right=False,
        width=1.0,
        colors=FRAME_COLOR,
    )


def _plot(results: dict) -> None:
    _configure_plotting()
    figure = plt.figure(figsize=(7.4, 5.4), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.25))
    axis_a = figure.add_subplot(grid[0, 0])
    axis_c = figure.add_subplot(grid[0, 1])
    axis_b = figure.add_subplot(grid[1, :])

    references = results["wu_references"]
    binary = results["deterministic_binary"]
    values = np.asarray(
        [
            references["passive_amplitude"],
            references["continuous_2omega_amplitude"],
            binary["nominal"]["amplitude"],
            results["landscape"]["neutral_amplitude"],
        ]
    )
    labels = [
        "Passive",
        "Continuous\n$2\\omega$",
        "Binary\n$2\\omega$",
        "Stochastic\nneutral",
    ]
    colors = (
        PASSIVE_COLOR,
        CONTINUOUS_COLOR,
        BINARY_COLOR,
        STOCHASTIC_COLOR,
    )
    x_values = np.arange(len(values))
    axis_a.plot(x_values, values, color="#A7ADB4", linewidth=1.2, zorder=1)
    axis_a.scatter(
        x_values,
        values,
        c=colors,
        s=52,
        edgecolor="white",
        linewidth=0.7,
        zorder=2,
    )
    axis_a.set(
        xticks=x_values,
        xticklabels=labels,
        ylabel="Steady amplitude",
        xlim=(-0.35, 3.35),
    )

    points = results["landscape"]["points"]
    nonneutral = [point for point in points if point["magnitude"] > 0.0]
    changes = np.asarray(
        [point["relative_change_percent"] for point in nonneutral]
    )
    color_limit = max(float(np.max(np.abs(changes))), 1e-12)
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
    scatter = axis_b.scatter(
        [point["phase_fraction"] for point in nonneutral],
        [point["magnitude"] for point in nonneutral],
        c=changes,
        cmap="RdBu_r",
        norm=norm,
        marker="s",
        s=115,
        linewidth=0.35,
        edgecolor="white",
    )
    axis_b.scatter(
        [0.0],
        [0.0],
        c=[0.0],
        cmap="RdBu_r",
        norm=norm,
        marker="D",
        s=45,
        linewidth=0.5,
        edgecolor=FRAME_COLOR,
        zorder=3,
    )
    best = results["landscape"]["best_point"]
    best_phase_fraction = (
        0.0 if best["phase_fraction"] is None else best["phase_fraction"]
    )
    axis_b.scatter(
        [best_phase_fraction],
        [best["magnitude"]],
        marker="*",
        s=95,
        facecolor="#F2C14E",
        edgecolor=FRAME_COLOR,
        linewidth=0.7,
        zorder=4,
    )
    axis_b.set(
        xlabel=r"Phase $\phi/2\pi$",
        ylabel=r"Magnitude $R$",
        xlim=(-0.035, 1.0),
        ylim=(-0.08, 1.10),
        xticks=np.linspace(0.0, 1.0, 5),
        yticks=[0.0, 0.25, 0.50, 1.00],
    )
    colorbar = figure.colorbar(scatter, ax=axis_b, pad=0.015, fraction=0.03)
    colorbar.set_label("Change from neutral (%)", fontsize=9)
    colorbar.outline.set_edgecolor(FRAME_COLOR)
    colorbar.outline.set_linewidth(1.1)
    colorbar.ax.tick_params(
        direction="in", width=1.0, colors=FRAME_COLOR, labelsize=8
    )

    axis_c.axhline(0.0, color=PASSIVE_COLOR, linestyle="--", linewidth=1.1)
    for radius, color in zip(LANDSCAPE_RADII, RADIUS_COLORS, strict=True):
        radius_points = [
            point
            for point in points
            if np.isclose(point["magnitude"], radius)
        ]
        phase_fraction = np.asarray(
            [point["phase_fraction"] for point in radius_points]
        )
        relative_change = np.asarray(
            [point["relative_change_percent"] for point in radius_points]
        )
        axis_c.plot(
            np.append(phase_fraction, 1.0),
            np.append(relative_change, relative_change[0]),
            color=color,
            linewidth=1.7,
            marker="o",
            markersize=2.8,
            label=f"R={radius:.2g}",
        )
    axis_c.scatter(
        [best_phase_fraction],
        [best["relative_change_percent"]],
        marker="*",
        s=70,
        facecolor="#F2C14E",
        edgecolor=FRAME_COLOR,
        linewidth=0.6,
        zorder=4,
    )
    axis_c.set(
        xlabel=r"Phase $\phi/2\pi$",
        ylabel="Change from neutral (%)",
        xlim=(0.0, 1.0),
        xticks=np.linspace(0.0, 1.0, 5),
    )
    axis_c.legend(loc="best")

    for label, axis in (("a", axis_a), ("b", axis_b), ("c", axis_c)):
        _style_axis(axis)
        axis.text(
            -0.14,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
    figure.savefig(FIGURE_PATH, dpi=600, bbox_inches="tight")
    plt.close(figure)


def _print_results(results: dict) -> None:
    references = results["wu_references"]
    binary = results["deterministic_binary"]
    best = results["landscape"]["best_point"]
    confirmation = results["confirmation"]
    print("## Control authority")
    print(f"passive={references['passive_amplitude']:.16g}")
    print(
        "continuous_2omega="
        f"{references['continuous_2omega_amplitude']:.16g}"
    )
    print(f"binary_2omega={binary['nominal']['amplitude']:.16g}")
    print(f"stochastic_neutral={results['landscape']['neutral_amplitude']:.16g}")
    print(f"authority_retained_fraction={binary['authority_retained_fraction']:.16g}")
    print("## Binary local FRF")
    print(
        f"peak_ratio={binary['local_frf']['peak_ratio']:.6g} "
        f"peak_amplitude={binary['local_frf']['peak_amplitude']:.16g}"
    )
    print("## Stochastic landscape")
    print(f"best_q={best['q']}")
    print(f"best_magnitude={best['magnitude']:.16g}")
    print(f"best_phase={best['phase']}")
    print(f"best_amplitude={best['mean_amplitude']:.16g}")
    print(f"best_change_percent={best['relative_change_percent']:.16g}")
    print("## Independent confirmation")
    print(f"neutral={confirmation['neutral']['mean_amplitude']:.16g}")
    print(f"best_q={confirmation['best_q']['mean_amplitude']:.16g}")
    print(
        "best_q_change_percent="
        f"{confirmation['best_q_relative_change_percent']:.16g}"
    )
    print("## Diagnosis")
    print(f"Case {results['diagnosis']['case']}")
    print(results["diagnosis"]["reason"])
    print(RESULTS_PATH.resolve())
    print(FIGURE_PATH.resolve())


def main() -> int:
    omega, omega_ratio, references = _load_frozen_inputs()
    phase = references["continuous_2omega_phase"]
    binary_nominal, binary_local_frf = _binary_local_frf(omega, phase)

    time_step, forcing = single_tone_forcing(
        FORCING_AMPLITUDE, omega, DIAGNOSTIC_NUM_PERIODS
    )
    times = time_step * np.arange(1, NUM_STEPS + 1, dtype=np.float64)
    landscape, landscape_uniforms = _landscape(
        forcing,
        times,
        omega,
        time_step,
        references["stochastic_a3_fixed_neutral_amplitude"],
    )
    best_q = np.asarray(landscape["best_point"]["q"], dtype=np.float64)
    confirmation = _confirmation(
        best_q,
        forcing,
        times,
        omega,
        time_step,
        landscape_uniforms,
    )

    passive = references["passive_amplitude"]
    continuous = references["continuous_2omega_amplitude"]
    binary_amplitude = binary_nominal["amplitude"]
    continuous_authority = passive - continuous
    if not np.isfinite(continuous_authority) or continuous_authority <= 0.0:
        raise RuntimeError("Frozen continuous two-omega authority is invalid")
    binary_reduction = 100.0 * (passive - binary_amplitude) / passive
    continuous_reduction = 100.0 * (passive - continuous) / passive
    binary_nominal.update(
        {
            "improvement_vs_passive_percent": binary_reduction,
            "improvement_vs_stochastic_neutral_percent": float(
                100.0
                * (landscape["neutral_amplitude"] - binary_amplitude)
                / landscape["neutral_amplitude"]
            ),
        }
    )
    binary = {
        "phase": phase,
        "preload_low": PRELOAD_LOW,
        "preload_high": PRELOAD_HIGH,
        "contacts_share_command": True,
        "nominal": binary_nominal,
        "local_frf": binary_local_frf,
        "continuous_reduction_percent": continuous_reduction,
        "binary_reduction_percent": binary_reduction,
        "continuous_to_binary_loss_points": float(
            continuous_reduction - binary_reduction
        ),
        "authority_retained_fraction": float(
            (passive - binary_amplitude) / continuous_authority
        ),
    }
    diagnosis = _diagnosis(
        binary_amplitude, passive, landscape, confirmation
    )
    results = {
        "configuration": {
            "mesh": "32x4 QUAD4",
            "num_free_dofs": SYSTEM.num_free_dofs,
            "damping": DAMPING,
            "forcing_amplitude": FORCING_AMPLITUDE,
            "omega_r": omega,
            "omega_r_ratio": omega_ratio,
            "frozen_preload": REFERENCE_PRELOAD,
            "preload_low": PRELOAD_LOW,
            "preload_high": PRELOAD_HIGH,
            "markov_base_seed": MARKOV_BASE_SEED,
            "num_periods": DIAGNOSTIC_NUM_PERIODS,
            "steps_per_period": 100,
            "local_frf_ratios": LOCAL_FRF_RATIOS.tolist(),
            "landscape_radii": [0.0, *LANDSCAPE_RADII.tolist()],
            "landscape_phases_per_nonzero_radius": 16,
            "landscape_stream": LANDSCAPE_STREAM,
            "confirmation_stream": CONFIRMATION_STREAM,
            "realizations_per_bank": 64,
            "all_realizations_share_nominal_forcing": True,
        },
        "wu_references": references,
        "deterministic_binary": binary,
        "landscape": landscape,
        "confirmation": confirmation,
        "phase_comparison": {
            "deterministic_2omega_phase": phase,
            "stochastic_best_q_phase": landscape["best_point"]["phase"],
            "phases_are_not_required_to_match": True,
        },
        "diagnosis": diagnosis,
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    _plot(results)
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
