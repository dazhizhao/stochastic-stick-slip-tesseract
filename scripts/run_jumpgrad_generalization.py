"""Evaluate the frozen J1 controller on fresh held-out Markov tapes."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from stochastic_stick_slip.jumpgrad import (
    HELD_OUT_CONDITIONS,
    build_jumpgrad_controller,
    condition_descriptors,
    deterministic_condition_objectives,
    evaluate_jumpgrad_bank,
    flatten_jumpgrad_parameters,
    functional_jumpgrad_controller,
    jumpgrad_uniform_bank,
)


STREAM_ID = 13
ITERATION = 0
NUM_REALIZATIONS = 128
BATCH_SIZE = 32
SOURCE_COMMIT = "894c074"
OUTPUT_PATH = ROOT / "outputs/jumpgrad_generalization/results.json"

# Exact little-endian Float64 bytes of training.final_theta from J1 results.json.
FROZEN_FINAL_THETA_BASE64 = (
    "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"
)


def frozen_final_theta() -> np.ndarray:
    theta = np.frombuffer(
        base64.b64decode(FROZEN_FINAL_THETA_BASE64), dtype="<f8"
    ).copy()
    if theta.shape != (354,) or not np.all(np.isfinite(theta)):
        raise ValueError("embedded J1 final theta is invalid")
    return theta


def controller_outputs(theta: np.ndarray) -> np.ndarray:
    controller = build_jumpgrad_controller()
    descriptors = torch.from_numpy(condition_descriptors(HELD_OUT_CONDITIONS))
    with torch.no_grad():
        q = functional_jumpgrad_controller(
            controller, torch.from_numpy(theta), descriptors
        )
    return np.asarray(q, dtype=np.float64)


def summarize_rows(normalized: np.ndarray) -> list[dict[str, float]]:
    return [
        {
            "mean_normalized_response": float(np.mean(row)),
            "population_std_normalized_response": float(np.std(row)),
            "q05_normalized_response": float(np.quantile(row, 0.05)),
            "q95_normalized_response": float(np.quantile(row, 0.95)),
            "mean_reduction_percent": float(100.0 * (1.0 - np.mean(row))),
            "q05_reduction_percent": float(100.0 * (1.0 - np.quantile(row, 0.95))),
            "q95_reduction_percent": float(100.0 * (1.0 - np.quantile(row, 0.05))),
        }
        for row in normalized
    ]


def summarize_vector(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "population_std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def evaluate_in_batches(q: np.ndarray, tapes: np.ndarray) -> np.ndarray:
    batches = []
    for start in range(0, NUM_REALIZATIONS, BATCH_SIZE):
        stop = start + BATCH_SIZE
        batch = evaluate_jumpgrad_bank(
            q, HELD_OUT_CONDITIONS, tapes[:, start:stop]
        )["trajectory_objectives"]
        if batch.shape != (len(HELD_OUT_CONDITIONS), BATCH_SIZE):
            raise AssertionError("unexpected trajectory batch shape")
        batches.append(batch)
    return np.concatenate(batches, axis=1)


def main() -> None:
    started = time.perf_counter()
    controller = build_jumpgrad_controller()
    initial_theta = np.asarray(
        flatten_jumpgrad_parameters(controller).detach(), dtype=np.float64
    )
    final_theta = frozen_final_theta()
    initial_q = controller_outputs(initial_theta)
    trained_q = controller_outputs(final_theta)
    passive = deterministic_condition_objectives(
        HELD_OUT_CONDITIONS, "passive"
    )
    wu = deterministic_condition_objectives(
        HELD_OUT_CONDITIONS, "wu_continuous_2omega"
    )
    tapes = jumpgrad_uniform_bank(
        len(HELD_OUT_CONDITIONS), NUM_REALIZATIONS, STREAM_ID, ITERATION
    )

    initial_raw = evaluate_in_batches(initial_q, tapes)
    trained_raw = evaluate_in_batches(trained_q, tapes)
    initial_normalized = initial_raw / passive[:, None]
    trained_normalized = trained_raw / passive[:, None]
    initial_aggregate = 100.0 * (1.0 - np.mean(initial_normalized, axis=0))
    trained_aggregate = 100.0 * (1.0 - np.mean(trained_normalized, axis=0))
    paired_delta = trained_aggregate - initial_aggregate
    finite = all(
        np.all(np.isfinite(value))
        for value in (
            initial_q,
            trained_q,
            passive,
            wu,
            initial_normalized,
            trained_normalized,
            paired_delta,
        )
    )
    if not finite:
        raise FloatingPointError("fresh-seed generalization output is non-finite")

    result = {
        "configuration": {
            "source_controller_commit": SOURCE_COMMIT,
            "stream_id": STREAM_ID,
            "iteration": ITERATION,
            "num_conditions": int(len(HELD_OUT_CONDITIONS)),
            "num_realizations": NUM_REALIZATIONS,
            "batch_size": BATCH_SIZE,
            "condition_order": "amplitude-major",
            "normalization": "trajectory amplitude / deterministic passive condition amplitude",
            "aggregate_score": "mean over 8 conditions of reduction vs passive percent",
        },
        "conditions": [
            {
                "index": index,
                "forcing_ratio": float(condition[0]),
                "frequency_ratio": float(condition[1]),
            }
            for index, condition in enumerate(HELD_OUT_CONDITIONS)
        ],
        "references": {
            "passive_objectives": passive.tolist(),
            "wu2019_objectives": wu.tolist(),
            "wu2019_reduction_percent": (
                100.0 * (1.0 - wu / passive)
            ).tolist(),
        },
        "controller": {
            "initial_theta": initial_theta.tolist(),
            "frozen_final_theta": final_theta.tolist(),
            "initial_q": initial_q.tolist(),
            "trained_q": trained_q.tolist(),
        },
        "normalized_responses": {
            "initial": initial_normalized.tolist(),
            "trained": trained_normalized.tolist(),
        },
        "per_condition": {
            "initial": summarize_rows(initial_normalized),
            "trained": summarize_rows(trained_normalized),
        },
        "aggregate_reduction_percent": {
            "initial": initial_aggregate.tolist(),
            "trained": trained_aggregate.tolist(),
            "initial_summary": summarize_vector(initial_aggregate),
            "trained_summary": summarize_vector(trained_aggregate),
        },
        "paired_trained_minus_initial_percent": {
            "values": paired_delta.tolist(),
            "summary": summarize_vector(paired_delta),
            "improved_count": int(np.count_nonzero(paired_delta > 0.0)),
            "improved_fraction": float(np.mean(paired_delta > 0.0)),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "finite": True,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n")

    initial_summary = result["aggregate_reduction_percent"]["initial_summary"]
    trained_summary = result["aggregate_reduction_percent"]["trained_summary"]
    paired = result["paired_trained_minus_initial_percent"]
    print("jumpgrad_generalization=COMPLETE")
    print(f"initial_mean_reduction_percent={initial_summary['mean']:.12g}")
    print(f"trained_mean_reduction_percent={trained_summary['mean']:.12g}")
    print(f"initial_median_reduction_percent={initial_summary['median']:.12g}")
    print(f"trained_median_reduction_percent={trained_summary['median']:.12g}")
    print(f"paired_improved={paired['improved_count']}/{NUM_REALIZATIONS}")
    print(f"runtime_seconds={result['runtime_seconds']:.3f}")
    print(f"output={OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
