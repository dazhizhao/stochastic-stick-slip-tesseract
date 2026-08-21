"""Optimize the deterministic 50%-duty Wu-V2 binary phase comparator."""

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
    HARMONIC_PHASES,
    LOCAL_FRF_RATIOS,
)
from stochastic_stick_slip.wu_v2 import (
    DAMPING,
    DIAGNOSTIC_NUM_PERIODS,
    FORCING_AMPLITUDE,
    REFERENCE_PRELOAD,
    SYSTEM,
    diagnostic_steady_state_metrics,
    simulate_preload_bank,
)
from stochastic_stick_slip.wu_v2_markov import (
    PRELOAD_HIGH,
    PRELOAD_LOW,
    deterministic_binary_preload,
    deterministic_policy_hard_limit_preload,
)


W1_PATH = ROOT / "outputs/wu2019_reproduction/scorecard.json"
W2_PATH = ROOT / "outputs/wu_v2_head_to_head/results.json"
OUTPUT_DIRECTORY = ROOT / "outputs/wu_v2_binary_comparator"
RESULTS_PATH = OUTPUT_DIRECTORY / "results.json"
MARKDOWN_PATH = OUTPUT_DIRECTORY / "binary_comparator.md"
FIGURE_PATH = OUTPUT_DIRECTORY / "binary_comparator.png"

BINARY_PHASES = HARMONIC_PHASES.copy()
WU_PHASE_INDEX = 46
CANDIDATE_NAMES = ("stochastic_lr0p1", "stochastic_lr1p0")
EXPECTED_Q = {
    "stochastic_lr0p1": np.asarray(
        [-3.514982271973227, -0.5337623522652377], dtype=np.float64
    ),
    "stochastic_lr1p0": np.asarray(
        [-10.665739565561044, -6.033414703985564], dtype=np.float64
    ),
}
EXPECTED_COEFFICIENT_PHASE = {
    "stochastic_lr0p1": 3.29229481812361,
    "stochastic_lr1p0": 3.656395858852916,
}
EXPECTED_BINARY_PHASE = {
    "stochastic_lr0p1": 4.561686815850873,
    "stochastic_lr1p0": 4.197585775121567,
}
EXPECTED_PASSIVE_PEAK = 0.18748720511761083
EXPECTED_WU_CONTINUOUS_PEAK = 0.1495599768466055
EXPECTED_WU_BINARY_PEAK = 0.14913023166130196
EXPECTED_STOCHASTIC_PEAK = {
    "stochastic_lr0p1": 0.14472949674894187,
    "stochastic_lr1p0": 0.14303627505618013,
}
REFERENCE_RTOL = 1e-12
REFERENCE_ATOL = 1e-14
VERY_CLOSE_PERCENT = 1.0

FRAME_COLOR = "#20242A"
LANDSCAPE_COLOR = "#376A8B"
WU_COLOR = "#315F55"
WU_BINARY_COLOR = "#527E99"
OPTIMIZED_COLOR = "#2F746F"
LR01_COLOR = "#7D5687"
LR10_COLOR = "#B36A4C"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _allclose(left, right) -> bool:
    return bool(
        np.allclose(left, right, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL)
    )


def coefficient_phase_to_binary_phase(coefficient_phase: float) -> float:
    """Convert h=R*cos(2wt-phi_q) to sin(2wt+phi_binary)."""
    return float(np.mod(0.5 * np.pi - coefficient_phase, 2.0 * np.pi))


def load_frozen_inputs() -> dict:
    w1 = _read_json(W1_PATH)
    w2 = _read_json(W2_PATH)
    configuration = w2["configuration"]
    references = w2["frozen_references"]
    comparisons = w2["comparisons"]["methods"]
    w1_two = w1["harmonic_search"]["2"]
    frozen = (
        w1["interpretation"]["category"] == "Partial reproduction"
        and _allclose(configuration["frequency_ratios"], LOCAL_FRF_RATIOS)
        and _allclose(configuration["preload_low"], PRELOAD_LOW)
        and _allclose(configuration["preload_high"], PRELOAD_HIGH)
        and configuration["num_periods"] == DIAGNOSTIC_NUM_PERIODS
        and configuration["steps_per_period"] == 100
        and _allclose(configuration["damping"], DAMPING)
        and _allclose(configuration["forcing_amplitude"], FORCING_AMPLITUDE)
        and _allclose(references["passive_preload"], REFERENCE_PRELOAD)
        and _allclose(references["passive_local_peak"], EXPECTED_PASSIVE_PEAK)
        and _allclose(
            references["wu_2omega_local_peak"], EXPECTED_WU_CONTINUOUS_PEAK
        )
        and _allclose(
            comparisons["binary_deterministic_2omega"]["local_peak_amplitude"],
            EXPECTED_WU_BINARY_PEAK,
        )
        and _allclose(w1_two["best_phase_rad"], BINARY_PHASES[WU_PHASE_INDEX])
        and _allclose(
            references["wu_2omega_phase"], BINARY_PHASES[WU_PHASE_INDEX]
        )
    )
    if not frozen:
        raise RuntimeError("W1 or W2 deterministic references do not match W3")

    candidates = {}
    for name in CANDIDATE_NAMES:
        source = w2["candidates"][name]
        q = np.asarray(source["q"], dtype=np.float64)
        coefficient_phase = float(source["coefficient_phase"])
        binary_phase = coefficient_phase_to_binary_phase(coefficient_phase)
        valid = (
            _allclose(q, EXPECTED_Q[name])
            and _allclose(coefficient_phase, EXPECTED_COEFFICIENT_PHASE[name])
            and _allclose(binary_phase, EXPECTED_BINARY_PHASE[name])
            and _allclose(
                comparisons[name]["local_peak_amplitude"],
                EXPECTED_STOCHASTIC_PEAK[name],
            )
            and comparisons[name]["range_status"] == "interior"
        )
        if not valid:
            raise RuntimeError(f"Frozen W2 candidate mismatch: {name}")
        candidates[name] = {
            "q": q,
            "coefficient_phase": coefficient_phase,
            "binary_phase": binary_phase,
            "stochastic_local_peak": float(
                comparisons[name]["local_peak_amplitude"]
            ),
        }
    return {
        "w1": w1,
        "w2": w2,
        "omega_r": float(configuration["omega_r"]),
        "omega_r_ratio": float(configuration["omega_r_ratio"]),
        "frequency_ratios": np.asarray(
            configuration["frequency_ratios"], dtype=np.float64
        ),
        "passive_peak": float(references["passive_local_peak"]),
        "wu_continuous_peak": float(references["wu_2omega_local_peak"]),
        "wu_continuous_phase": float(references["wu_2omega_phase"]),
        "wu_binary_peak": float(
            comparisons["binary_deterministic_2omega"][
                "local_peak_amplitude"
            ]
        ),
        "candidates": candidates,
    }


def select_phase_by_local_peak(
    steady_amplitudes: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Select the first phase minimizing its maximum across frequency."""
    values = np.asarray(steady_amplitudes, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("phase FRF amplitudes must have shape (phase, frequency)")
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("phase FRF amplitudes must be finite")
    local_peaks = np.max(values, axis=1)
    peak_frequency_indices = np.argmax(values, axis=1)
    return int(np.argmin(local_peaks)), local_peaks, peak_frequency_indices


def _local_frf_summary(
    ratios: np.ndarray,
    amplitudes: np.ndarray,
    steady_errors: np.ndarray,
) -> dict:
    ratios = np.asarray(ratios, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    steady_errors = np.asarray(steady_errors, dtype=np.float64)
    if (
        amplitudes.shape != ratios.shape
        or steady_errors.shape != ratios.shape
        or not np.all(np.isfinite(amplitudes))
        or not np.all(np.isfinite(steady_errors))
    ):
        raise FloatingPointError("local FRF data are invalid")
    peak_index = int(np.argmax(amplitudes))
    boundary = peak_index in (0, len(ratios) - 1)
    return {
        "steady_amplitudes": amplitudes.tolist(),
        "steady_errors": steady_errors.tolist(),
        "peak_index": peak_index,
        "peak_ratio": float(ratios[peak_index]),
        "peak_amplitude": float(amplitudes[peak_index]),
        "peak_steady_error": float(steady_errors[peak_index]),
        "peak_at_boundary": boundary,
        "range_status": "range_insufficient" if boundary else "interior",
    }


def _run_deterministic_sweep(frozen: dict) -> tuple[dict, float]:
    started = time.perf_counter()
    ratios = frozen["frequency_ratios"]
    phase_amplitudes = np.empty((len(BINARY_PHASES), len(ratios)))
    phase_errors = np.empty_like(phase_amplitudes)
    hard_amplitudes = np.empty((len(CANDIDATE_NAMES), len(ratios)))
    hard_errors = np.empty_like(hard_amplitudes)
    q_bank = np.stack(
        [frozen["candidates"][name]["q"] for name in CANDIDATE_NAMES]
    )

    for frequency_index, ratio in enumerate(ratios):
        omega = float(ratio * frozen["omega_r"])
        phase_preload = deterministic_binary_preload(omega, BINARY_PHASES)
        hard_preload = deterministic_policy_hard_limit_preload(omega, q_bank)
        preload = np.concatenate((phase_preload, hard_preload), axis=0)
        displacement = np.asarray(simulate_preload_bank(omega, preload)[0])
        objective, steady_error, _ = diagnostic_steady_state_metrics(displacement)
        if not np.all(np.isfinite(objective)) or not np.all(
            np.isfinite(steady_error)
        ):
            raise FloatingPointError(
                f"non-finite deterministic result at frequency ratio {ratio}"
            )
        phase_amplitudes[:, frequency_index] = objective[: len(BINARY_PHASES)]
        phase_errors[:, frequency_index] = steady_error[: len(BINARY_PHASES)]
        hard_amplitudes[:, frequency_index] = objective[len(BINARY_PHASES) :]
        hard_errors[:, frequency_index] = steady_error[len(BINARY_PHASES) :]
        print(f"binary_comparator_frf={frequency_index + 1:02d}/{len(ratios)}")

    best_index, local_peaks, peak_indices = select_phase_by_local_peak(
        phase_amplitudes
    )
    peak_errors = phase_errors[np.arange(len(BINARY_PHASES)), peak_indices]
    boundary = np.isin(peak_indices, [0, len(ratios) - 1])
    if not np.isclose(
        local_peaks[WU_PHASE_INDEX],
        frozen["wu_binary_peak"],
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    ):
        raise AssertionError("Wu-phase binary no longer reproduces W2")

    optimized = _local_frf_summary(
        ratios, phase_amplitudes[best_index], phase_errors[best_index]
    )
    optimized.update(
        {
            "phase_index": best_index,
            "phase": float(BINARY_PHASES[best_index]),
            "phase_fraction": float(BINARY_PHASES[best_index] / (2.0 * np.pi)),
        }
    )
    hard_limits = {}
    for index, name in enumerate(CANDIDATE_NAMES):
        summary = _local_frf_summary(
            ratios, hard_amplitudes[index], hard_errors[index]
        )
        summary.update(
            {
                "q": frozen["candidates"][name]["q"].tolist(),
                "coefficient_phase": frozen["candidates"][name][
                    "coefficient_phase"
                ],
                "equivalent_binary_phase": frozen["candidates"][name][
                    "binary_phase"
                ],
                "equivalent_binary_phase_fraction": float(
                    frozen["candidates"][name]["binary_phase"] / (2.0 * np.pi)
                ),
            }
        )
        hard_limits[name] = summary

    return (
        {
            "phase_amplitudes": phase_amplitudes,
            "phase_errors": phase_errors,
            "local_peaks": local_peaks,
            "peak_indices": peak_indices,
            "peak_errors": peak_errors,
            "peak_at_boundary": boundary,
            "optimized": optimized,
            "hard_limits": hard_limits,
        },
        time.perf_counter() - started,
    )


def _margin(deterministic: float, stochastic: float) -> dict:
    absolute = float(deterministic - stochastic)
    relative = float(100.0 * absolute / deterministic)
    if abs(relative) < VERY_CLOSE_PERCENT:
        descriptor = "very_close_within_one_percent"
    elif absolute > 0.0:
        descriptor = "stochastic_lower"
    elif absolute < 0.0:
        descriptor = "deterministic_lower"
    else:
        descriptor = "exact_tie"
    return {
        "definition": "deterministic minus stochastic; positive favors stochastic",
        "deterministic_amplitude": float(deterministic),
        "stochastic_amplitude": float(stochastic),
        "absolute_margin": absolute,
        "relative_margin_percent": relative,
        "within_one_percent": bool(abs(relative) < VERY_CLOSE_PERCENT),
        "descriptor": descriptor,
    }


def _comparison_table(frozen: dict, sweep: dict) -> dict:
    passive = frozen["passive_peak"]
    wu = frozen["wu_continuous_peak"]
    methods = {
        "passive": passive,
        "wu_continuous_2omega": wu,
        "wu_phase_binary": frozen["wu_binary_peak"],
        "phase_optimized_binary": sweep["optimized"]["peak_amplitude"],
        "learned_hard_limit_lr0p1": sweep["hard_limits"][
            "stochastic_lr0p1"
        ]["peak_amplitude"],
        "stochastic_lr0p1": frozen["candidates"]["stochastic_lr0p1"][
            "stochastic_local_peak"
        ],
        "learned_hard_limit_lr1p0": sweep["hard_limits"][
            "stochastic_lr1p0"
        ]["peak_amplitude"],
        "stochastic_lr1p0": frozen["candidates"]["stochastic_lr1p0"][
            "stochastic_local_peak"
        ],
    }
    return {
        name: {
            "local_peak_amplitude": float(amplitude),
            "reduction_vs_passive_percent": float(
                100.0 * (passive - amplitude) / passive
            ),
            "improvement_vs_wu_percent": float(100.0 * (wu - amplitude) / wu),
        }
        for name, amplitude in methods.items()
    }


def _interpret(frozen: dict, sweep: dict, margins: dict) -> dict:
    optimized = sweep["optimized"]
    hard_lr1 = sweep["hard_limits"]["stochastic_lr1p0"]
    stochastic_lr01 = frozen["candidates"]["stochastic_lr0p1"][
        "stochastic_local_peak"
    ]
    stochastic_lr10 = frozen["candidates"]["stochastic_lr1p0"][
        "stochastic_local_peak"
    ]
    if optimized["peak_at_boundary"] or hard_lr1["peak_at_boundary"]:
        category = "range_insufficient"
        statement = (
            "The sampled local window is insufficient for the decisive binary "
            "comparison; no Case A/B/C claim is made."
        )
    elif optimized["peak_amplitude"] < min(stochastic_lr01, stochastic_lr10):
        category = "Case C"
        statement = (
            "The phase-optimized deterministic binary controller is lower than "
            "both frozen stochastic policies on this benchmark."
        )
    elif (
        stochastic_lr10 < optimized["peak_amplitude"]
        and stochastic_lr10 < hard_lr1["peak_amplitude"]
    ):
        category = "Case A"
        statement = (
            "The optimized stochastic Markov policy provides additional "
            "performance beyond the phase-optimized deterministic 50%-duty "
            "binary 2ω controller on the same benchmark."
        )
    else:
        category = "Case B"
        statement = (
            "The current data do not isolate a finite-rate stochastic advantage; "
            "CRN-FD is consistent with discovering an effective phase-locked "
            "hard-switching law."
        )
    if margins["optimized_binary_minus_stochastic_lr1p0"][
        "within_one_percent"
    ]:
        statement += (
            " The optimized-binary and stochastic lr=1.0 peaks are within "
            "1% and therefore very close."
        )
    return {
        "category": category,
        "statement": statement,
        "signed_margins_are_primary": True,
        "descriptive_close_band_percent": VERY_CLOSE_PERCENT,
        "margins": margins,
        "claim_boundary": (
            "This is a comparator result on the same JAX-FEM benchmark, not a "
            "claim that randomness is universally beneficial or that Wu et al. "
            "2019 is outperformed."
        ),
    }


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.8,
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


def _plot(results: dict) -> None:
    _configure_plotting()
    passive = results["frozen_references"]["passive_local_peak"]
    landscape = results["phase_landscape"]
    phase_fraction = np.asarray(landscape["phase_fraction"])
    reductions = np.asarray(landscape["reduction_vs_passive_percent"])
    closed_phase = np.concatenate((phase_fraction, [1.0]))
    closed_reduction = np.concatenate((reductions, reductions[:1]))
    optimized = results["optimized_binary"]
    hard_limits = results["learned_phase_hard_limits"]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.25),
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    axis = axes[0]
    axis.plot(
        closed_phase,
        closed_reduction,
        color=LANDSCAPE_COLOR,
        linewidth=1.7,
        marker="o",
        markersize=2.7,
        label="64-phase binary grid",
    )
    markers = [
        (
            results["frozen_references"]["wu_binary_phase_fraction"],
            results["comparisons"]["wu_phase_binary"][
                "reduction_vs_passive_percent"
            ],
            "s",
            WU_BINARY_COLOR,
            "Wu-phase binary",
        ),
        (
            optimized["phase_fraction"],
            results["comparisons"]["phase_optimized_binary"][
                "reduction_vs_passive_percent"
            ],
            "*",
            OPTIMIZED_COLOR,
            "Optimized binary",
        ),
        (
            hard_limits["stochastic_lr0p1"]["equivalent_binary_phase_fraction"],
            100.0
            * (
                passive
                - hard_limits["stochastic_lr0p1"]["peak_amplitude"]
            )
            / passive,
            "D",
            LR01_COLOR,
            "lr=0.1 hard limit",
        ),
        (
            hard_limits["stochastic_lr1p0"]["equivalent_binary_phase_fraction"],
            100.0
            * (
                passive
                - hard_limits["stochastic_lr1p0"]["peak_amplitude"]
            )
            / passive,
            "^",
            LR10_COLOR,
            "lr=1.0 hard limit",
        ),
    ]
    for x, y, marker, color, label in markers:
        axis.scatter(
            [x],
            [y],
            marker=marker,
            s=55 if marker == "*" else 34,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=4,
            label=label,
        )
    axis.set_xlabel(r"Binary phase, $\phi/(2\pi)$")
    axis.set_ylabel("Peak reduction vs passive (%)")
    axis.set_xlim(0.0, 1.0)
    axis.legend(loc="upper left", ncol=1, handletextpad=0.4)
    axis.text(-0.12, 1.03, "a", transform=axis.transAxes, fontweight="bold")
    _style_axis(axis)

    axis = axes[1]
    method_names = (
        "wu_continuous_2omega",
        "wu_phase_binary",
        "phase_optimized_binary",
        "stochastic_lr0p1",
        "stochastic_lr1p0",
    )
    values = [
        results["comparisons"][name]["reduction_vs_passive_percent"]
        for name in method_names
    ]
    colors = [WU_COLOR, WU_BINARY_COLOR, OPTIMIZED_COLOR, LR01_COLOR, LR10_COLOR]
    labels = ["Wu cont.\n2ω", "Wu-phase\nbinary", "Optimized\nbinary", "Stoch.\nlr=0.1", "Stoch.\nlr=1.0"]
    bars = axis.bar(np.arange(len(values)), values, color=colors, width=0.70)
    axis.set_xticks(np.arange(len(values)), labels)
    axis.set_ylabel("Peak reduction vs passive (%)")
    padding = 0.25
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + padding,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=FRAME_COLOR,
        )
    axis.text(-0.16, 1.03, "b", transform=axis.transAxes, fontweight="bold")
    _style_axis(axis)
    figure.tight_layout(w_pad=2.2)
    figure.savefig(FIGURE_PATH, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _write_markdown(results: dict) -> None:
    comparison = results["comparisons"]
    labels = {
        "passive": "Passive",
        "wu_continuous_2omega": "Wu continuous 2ω",
        "wu_phase_binary": "Wu-phase binary",
        "phase_optimized_binary": "Phase-optimized binary",
        "learned_hard_limit_lr0p1": "Learned hard limit, lr=0.1",
        "stochastic_lr0p1": "Stochastic lr=0.1",
        "learned_hard_limit_lr1p0": "Learned hard limit, lr=1.0",
        "stochastic_lr1p0": "Stochastic lr=1.0",
    }
    lines = [
        "# Phase-optimized deterministic Wu-V2 binary comparator",
        "",
        "All stochastic amplitudes are frozen from W2; W3 performs only "
        "deterministic forward evaluations.",
        "",
        "## Comparison",
        "",
        "| Method | Local peak | vs Passive | vs Wu continuous |",
        "|---|---:|---:|---:|",
    ]
    for name in labels:
        entry = comparison[name]
        versus_wu = (
            "—"
            if name == "wu_continuous_2omega"
            else f"{entry['improvement_vs_wu_percent']:.6f}%"
        )
        lines.append(
            f"| {labels[name]} | {entry['local_peak_amplitude']:.12f} | "
            f"{entry['reduction_vs_passive_percent']:.6f}% | {versus_wu} |"
        )

    lines.extend(["", "## Signed margins", ""])
    for name, margin in results["interpretation"]["margins"].items():
        descriptor = margin["descriptor"].replace("_", " ")
        lines.append(
            f"- {name}: `{margin['absolute_margin']:.12g}` "
            f"(`{margin['relative_margin_percent']:.6f}%`), "
            f"{descriptor}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"**{results['interpretation']['category']}**",
            "",
            results["interpretation"]["statement"],
            "",
            results["interpretation"]["claim_boundary"],
            "",
        ]
    )
    MARKDOWN_PATH.write_text("\n".join(lines))


def main() -> None:
    total_started = time.perf_counter()
    frozen = load_frozen_inputs()
    print("frozen_w1_w2_inputs=validated")
    sweep, sweep_seconds = _run_deterministic_sweep(frozen)
    comparisons = _comparison_table(frozen, sweep)
    margins = {
        "optimized_binary_minus_stochastic_lr1p0": _margin(
            sweep["optimized"]["peak_amplitude"],
            frozen["candidates"]["stochastic_lr1p0"]["stochastic_local_peak"],
        ),
        "hard_limit_lr0p1_minus_stochastic_lr0p1": _margin(
            sweep["hard_limits"]["stochastic_lr0p1"]["peak_amplitude"],
            frozen["candidates"]["stochastic_lr0p1"]["stochastic_local_peak"],
        ),
        "hard_limit_lr1p0_minus_stochastic_lr1p0": _margin(
            sweep["hard_limits"]["stochastic_lr1p0"]["peak_amplitude"],
            frozen["candidates"]["stochastic_lr1p0"]["stochastic_local_peak"],
        ),
    }
    interpretation = _interpret(frozen, sweep, margins)
    reductions = 100.0 * (
        frozen["passive_peak"] - sweep["local_peaks"]
    ) / frozen["passive_peak"]
    results = {
        "configuration": {
            "mesh": "32x4 QUAD4",
            "num_free_dofs": SYSTEM.num_free_dofs,
            "damping": DAMPING,
            "forcing_amplitude": FORCING_AMPLITUDE,
            "preload_low": PRELOAD_LOW,
            "preload_high": PRELOAD_HIGH,
            "binary_duty_cycle": 0.5,
            "omega_r": frozen["omega_r"],
            "omega_r_ratio": frozen["omega_r_ratio"],
            "num_periods": DIAGNOSTIC_NUM_PERIODS,
            "steps_per_period": 100,
            "objective_cycles": [21, 22, 23, 24],
            "frequency_ratios": frozen["frequency_ratios"].tolist(),
            "phase_count": len(BINARY_PHASES),
            "phase_refinement_used": False,
            "stochastic_forwards_or_training_used": False,
        },
        "frozen_references": {
            "passive_local_peak": frozen["passive_peak"],
            "wu_continuous_local_peak": frozen["wu_continuous_peak"],
            "wu_binary_local_peak": frozen["wu_binary_peak"],
            "wu_binary_phase": frozen["wu_continuous_phase"],
            "wu_binary_phase_fraction": float(
                frozen["wu_continuous_phase"] / (2.0 * np.pi)
            ),
            "stochastic_lr0p1_local_peak": frozen["candidates"][
                "stochastic_lr0p1"
            ]["stochastic_local_peak"],
            "stochastic_lr1p0_local_peak": frozen["candidates"][
                "stochastic_lr1p0"
            ]["stochastic_local_peak"],
            "w2_stochastic_results_recomputed": False,
        },
        "phase_landscape": {
            "phases": BINARY_PHASES.tolist(),
            "phase_fraction": (BINARY_PHASES / (2.0 * np.pi)).tolist(),
            "local_peak_amplitudes": sweep["local_peaks"].tolist(),
            "reduction_vs_passive_percent": reductions.tolist(),
            "peak_frequency_indices": sweep["peak_indices"].tolist(),
            "peak_frequency_ratios": frozen["frequency_ratios"][
                sweep["peak_indices"]
            ].tolist(),
            "peak_steady_errors": sweep["peak_errors"].tolist(),
            "peak_at_boundary": sweep["peak_at_boundary"].tolist(),
            "selection_metric": "minimum over phase of max over 21 frequencies",
            "tie_breaking": "first phase in registered grid order",
            "plotting_closure_adds_scientific_sample": False,
        },
        "optimized_binary": sweep["optimized"],
        "learned_phase_hard_limits": sweep["hard_limits"],
        "comparisons": comparisons,
        "interpretation": interpretation,
        "runtime": {
            "deterministic_sweep_seconds": sweep_seconds,
            "batched_fem_calls": len(LOCAL_FRF_RATIOS),
            "deterministic_cases": len(LOCAL_FRF_RATIOS)
            * (len(BINARY_PHASES) + len(CANDIDATE_NAMES)),
            "stochastic_forward_calls": 0,
            "training_or_gradient_calls": 0,
        },
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    _write_markdown(results)
    _plot(results)
    results["runtime"]["total_runner_seconds"] = time.perf_counter() - total_started
    RESULTS_PATH.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")

    print("## W3 deterministic binary comparator")
    print(
        f"optimized_phase={sweep['optimized']['phase']:.16g} "
        f"local_peak={sweep['optimized']['peak_amplitude']:.16g} "
        f"peak_ratio={sweep['optimized']['peak_ratio']:.3f}"
    )
    for name, margin in margins.items():
        print(
            f"{name}: absolute={margin['absolute_margin']:.16g} "
            f"relative={margin['relative_margin_percent']:.9g}%"
        )
    print(f"interpretation={interpretation['category']}")
    print(f"results={RESULTS_PATH}")
    print(f"figure={FIGURE_PATH}")


if __name__ == "__main__":
    main()
