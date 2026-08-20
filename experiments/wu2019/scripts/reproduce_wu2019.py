"""Reproduce the numerical SDOF conclusions reported by Wu et al. (2019)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from wu2019.controller import constant_normal_force, harmonic_normal_force
from wu2019.dynamics import (
    DEFAULT_SETTINGS,
    SimulationSettings,
    dense_frequency_grid,
    display_frequency_grid,
    sweep_frequency_grid,
)
from wu2019.newmark import simulate_summary_batch


OUTPUT_DIRECTORY = EXPERIMENT_ROOT / "outputs"
SUMMARY_PATH = OUTPUT_DIRECTORY / "reproduction_summary.json"
CONSTANT_FIGURE_PATH = OUTPUT_DIRECTORY / "constant_preload_frf.png"
PUBLISHED_FIGURE_PATH = OUTPUT_DIRECTORY / "published_controller_frf.png"

CONSTANT_COLOR = "#555B63"
WU_COLOR = "#27628D"
FOURTH_COLOR = "#C47A32"
FIRST_COLOR = "#8A8F98"
FRAME_COLOR = "#20242A"


def _configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _style_axis(axis) -> None:
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
    axis.grid(False)


def _frf(omegas, normal_force, settings=DEFAULT_SETTINGS):
    return simulate_summary_batch(omegas, normal_force, settings)


def _peak(omegas, amplitude):
    index = int(np.argmax(amplitude))
    return {
        "omega": float(omegas[index]),
        "amplitude_m": float(amplitude[index]),
        "amplitude_mm": float(1e3 * amplitude[index]),
    }


def _reduction(reference, candidate):
    return 100.0 * (reference - candidate) / reference


def _plot_constant_frfs(omegas, curves) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    colors = ["#A9ADB2", "#6E91AD", "#27628D", "#C47A32", "#7D5B78"]
    for (normal_force, amplitude), color in zip(curves.items(), colors, strict=True):
        axis.plot(
            omegas,
            1e3 * amplitude,
            color=color,
            linewidth=1.8,
            label=f"N = {normal_force:g} N",
        )
    axis.set_xlabel("Excitation frequency (rad s$^{-1}$)")
    axis.set_ylabel("Response amplitude (mm)")
    axis.legend(loc="upper right")
    _style_axis(axis)
    figure.savefig(CONSTANT_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_published_frfs(omegas, curves) -> None:
    figure, axis = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    styles = {
        "Constant N = 40 N": (CONSTANT_COLOR, 2.0, "-"),
        "Wu 2$\\omega$": (WU_COLOR, 2.2, "-"),
        "Wu 4$\\omega$": (FOURTH_COLOR, 1.8, "--"),
        "Near-best 1$\\omega$": (FIRST_COLOR, 1.6, ":"),
    }
    for label, amplitude in curves.items():
        color, width, linestyle = styles[label]
        axis.plot(
            omegas,
            1e3 * amplitude,
            color=color,
            linewidth=width,
            linestyle=linestyle,
            label=label,
        )
    axis.set_xlabel("Excitation frequency (rad s$^{-1}$)")
    axis.set_ylabel("Response amplitude (mm)")
    axis.legend(loc="upper right")
    _style_axis(axis)
    figure.savefig(PUBLISHED_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    _configure_plotting()
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    settings = DEFAULT_SETTINGS
    start = time.perf_counter()

    display_omegas = display_frequency_grid()
    constant_curves = {}
    constant_summaries = {}
    for normal_force in (0.0, 20.0, 40.0, 100.0, 1000.0):
        summary = _frf(
            display_omegas,
            constant_normal_force(normal_force, settings),
        )
        constant_curves[normal_force] = summary.amplitude
        constant_summaries[str(int(normal_force))] = _peak(
            display_omegas, summary.amplitude
        )
        print(
            f"constant N={normal_force:g} "
            f"peak={constant_summaries[str(int(normal_force))]['amplitude_mm']:.6g} mm",
            flush=True,
        )

    sweep_omegas = sweep_frequency_grid()
    preload_values = np.arange(0.0, 100.0 + 1e-12, 5.0)
    preload_peaks = []
    for normal_force in preload_values:
        summary = _frf(
            sweep_omegas,
            constant_normal_force(normal_force, settings),
        )
        preload_peaks.append(float(np.max(summary.amplitude)))
    preload_peaks = np.asarray(preload_peaks)
    optimum_index = int(np.argmin(preload_peaks))
    optimum_preload = float(preload_values[optimum_index])

    dense_omegas = dense_frequency_grid()
    constant_history = constant_normal_force(40.0, settings)
    wu_history = harmonic_normal_force(40.0, 10.0, 2, 4.4, settings)
    fourth_history = harmonic_normal_force(40.0, 5.0, 4, 5.0, settings)
    constant_dense = _frf(dense_omegas, constant_history)
    wu_dense = _frf(dense_omegas, wu_history)
    fourth_dense = _frf(dense_omegas, fourth_history)

    phase_candidates = np.arange(16, dtype=np.float64) * np.pi / 8.0
    first_candidates = []
    for phase in phase_candidates:
        result = _frf(
            dense_omegas,
            harmonic_normal_force(40.0, 10.0, 1, phase, settings),
        )
        first_candidates.append(result)
    first_peaks = np.asarray(
        [np.max(result.amplitude) for result in first_candidates]
    )
    first_index = int(np.argmin(first_peaks))
    first_phase = float(phase_candidates[first_index])
    first_dense = first_candidates[first_index]

    constant_peak = _peak(dense_omegas, constant_dense.amplitude)
    wu_peak = _peak(dense_omegas, wu_dense.amplitude)
    fourth_peak = _peak(dense_omegas, fourth_dense.amplitude)
    first_peak = _peak(dense_omegas, first_dense.amplitude)
    wu_reduction = _reduction(
        constant_peak["amplitude_m"], wu_peak["amplitude_m"]
    )
    fourth_reduction = _reduction(
        constant_peak["amplitude_m"], fourth_peak["amplitude_m"]
    )
    first_reduction = _reduction(
        constant_peak["amplitude_m"], first_peak["amplitude_m"]
    )

    constant_peak_index = int(
        np.argmax(constant_dense.amplitude)
    )
    wu_peak_index = int(np.argmax(wu_dense.amplitude))
    convergence = {}
    for name, result, index in (
        ("constant", constant_dense, constant_peak_index),
        ("wu_2omega", wu_dense, wu_peak_index),
    ):
        previous = float(result.previous_ten_amplitude[index])
        final = float(result.last_ten_amplitude[index])
        convergence[name] = abs(final - previous) / max(abs(final), 1e-15)

    fine_settings = SimulationSettings(
        steps_per_period=800,
        num_periods=100,
        measurement_periods=20,
    )
    timestep = {}
    for name, omega, coarse_amplitude, control in (
        (
            "constant",
            constant_peak["omega"],
            constant_peak["amplitude_m"],
            constant_normal_force(40.0, fine_settings),
        ),
        (
            "wu_2omega",
            wu_peak["omega"],
            wu_peak["amplitude_m"],
            harmonic_normal_force(40.0, 10.0, 2, 4.4, fine_settings),
        ),
    ):
        fine = _frf(np.array([omega]), control, fine_settings)
        fine_amplitude = float(fine.amplitude[0])
        timestep[name] = {
            "coarse_amplitude_m": coarse_amplitude,
            "fine_amplitude_m": fine_amplitude,
            "relative_change": abs(fine_amplitude - coarse_amplitude)
            / max(abs(fine_amplitude), 1e-15),
        }

    gates = {
        "constant_optimum_near_40": 35.0 <= optimum_preload <= 45.0,
        "resonance_in_band": 190.0 <= constant_peak["omega"] <= 220.0,
        "wu_reduction_near_21": 16.0 <= wu_reduction <= 26.0,
        "fourth_positive_and_weaker": (
            fourth_reduction > 0.0
            and wu_reduction - fourth_reduction >= 5.0
        ),
        "steady_state": max(convergence.values()) < 0.005,
        "time_step": max(
            value["relative_change"] for value in timestep.values()
        ) < 0.01,
        "friction_dissipative": (
            float(constant_dense.dissipated_energy[constant_peak_index])
            >= -1e-10
            and float(wu_dense.dissipated_energy[wu_peak_index]) >= -1e-10
        ),
        "friction_bounded": (
            float(np.max(constant_dense.friction_excess)) <= 1e-9
            and float(np.max(wu_dense.friction_excess)) <= 1e-9
        ),
    }
    diagnostics = {
        "near_best_1omega_reduction_percent": first_reduction,
        "near_best_1omega_at_or_below_original_3_percent": (
            first_reduction <= 3.0
        ),
    }
    passed = bool(all(gates.values()))

    _plot_constant_frfs(display_omegas, constant_curves)
    _plot_published_frfs(
        dense_omegas,
        {
            "Constant N = 40 N": constant_dense.amplitude,
            "Wu 2$\\omega$": wu_dense.amplitude,
            "Wu 4$\\omega$": fourth_dense.amplitude,
            "Near-best 1$\\omega$": first_dense.amplitude,
        },
    )

    results = {
        "paper": {
            "doi": "10.1016/j.jsv.2019.114850",
            "reported_constant_optimum_N": 40.0,
            "reported_second_harmonic_reduction_percent": 21.0,
        },
        "settings": {
            "steps_per_period": settings.steps_per_period,
            "num_periods": settings.num_periods,
            "measurement_periods": settings.measurement_periods,
        },
        "constant_curves": constant_summaries,
        "constant_sweep": {
            "normal_force_N": preload_values.tolist(),
            "peak_amplitude_m": preload_peaks.tolist(),
            "optimum_normal_force_N": optimum_preload,
        },
        "published_comparison": {
            "constant": constant_peak,
            "wu_2omega": wu_peak,
            "wu_2omega_reduction_percent": wu_reduction,
            "wu_4omega": fourth_peak,
            "wu_4omega_reduction_percent": fourth_reduction,
            "near_best_1omega": first_peak,
            "near_best_1omega_phase_rad": first_phase,
            "near_best_1omega_reduction_percent": first_reduction,
        },
        "steady_state_relative_change": convergence,
        "time_step_check": timestep,
        "gates": gates,
        "diagnostics": diagnostics,
        "passed": passed,
        "runtime_seconds": time.perf_counter() - start,
        "outputs": {
            "constant_preload_frf": str(CONSTANT_FIGURE_PATH),
            "published_controller_frf": str(PUBLISHED_FIGURE_PATH),
        },
    }
    SUMMARY_PATH.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    print("\n## Wu 2019 reproduction", flush=True)
    print(f"constant optimum={optimum_preload:.6g} N", flush=True)
    print(
        f"constant peak={constant_peak['amplitude_mm']:.9g} mm "
        f"at {constant_peak['omega']:.6g} rad/s",
        flush=True,
    )
    print(
        f"Wu 2omega peak={wu_peak['amplitude_mm']:.9g} mm "
        f"reduction={wu_reduction:.6g}%",
        flush=True,
    )
    print(
        f"Wu 4omega reduction={fourth_reduction:.6g}%",
        flush=True,
    )
    print(
        f"near-best 1omega reduction={first_reduction:.6g}% "
        f"phase={first_phase:.6g} rad",
        flush=True,
    )
    print(f"gates={gates}", flush=True)
    print(f"diagnostics={diagnostics}", flush=True)
    print(f"PASS={passed}", flush=True)
    print(f"summary={SUMMARY_PATH}", flush=True)
    print(f"figure={CONSTANT_FIGURE_PATH}", flush=True)
    print(f"figure={PUBLISHED_FIGURE_PATH}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
