"""Compare Markov-jump CRN-FD gradients on two independent tape banks."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.run_markov_jump_gate_a import _centered_fd
from stochastic_stick_slip.engineering_markov import (
    BETA,
    DAMPING,
    FD_EPSILON,
    GATE_A_FORCING_SEEDS,
    LAMBDA_0,
    MARKOV_BASE_SEED,
    MARKOV_ITERATION,
    PRELOAD_HIGH,
    PRELOAD_LOW,
    gate_a_forcing,
    markov_uniform_bank,
)
from stochastic_stick_slip.model import NUM_FOURIER_COEFFICIENTS, NUM_STEPS


OUTPUT_PATH = ROOT / "outputs/markov_jump_gate_b/results.json"
BANK_A_STREAM = 5
BANK_B_STREAM = 6
COSINE_THRESHOLD = 0.5
COEFFICIENT_NAMES = ("a0", "a1", "b1", "a2", "b2")


def _bank_result(stream_id, num_realizations, forcing, shared_coefficients):
    uniforms = markov_uniform_bank(num_realizations, stream_id=stream_id)
    result = _centered_fd(shared_coefficients, forcing, uniforms)
    return {
        "stream_id": stream_id,
        "gradient": result["gradient"],
        "l2_norm": result["l2_norm"],
        "linf_norm": result["linf_norm"],
        "finite": result["finite"],
        "numerical_zero": result["numerical_zero"],
        "mode_difference_counts": result["mode_difference_counts"],
        "jenkins_difference_counts": result["jenkins_difference_counts"],
    }


def _sign_table(gradient_a, gradient_b):
    rows = []
    for name, value_a, value_b in zip(
        COEFFICIENT_NAMES, gradient_a, gradient_b, strict=True
    ):
        same_sign = bool(
            (value_a > 0.0 and value_b > 0.0)
            or (value_a < 0.0 and value_b < 0.0)
        )
        rows.append(
            {
                "coefficient": name,
                "g_A": float(value_a),
                "g_B": float(value_b),
                "same_sign": same_sign,
            }
        )
    return rows


def _run_comparison(num_realizations, forcing, shared_coefficients):
    bank_a = _bank_result(
        BANK_A_STREAM, num_realizations, forcing, shared_coefficients
    )
    bank_b = _bank_result(
        BANK_B_STREAM, num_realizations, forcing, shared_coefficients
    )
    gradient_a = np.asarray(bank_a["gradient"], dtype=np.float64)
    gradient_b = np.asarray(bank_b["gradient"], dtype=np.float64)
    gradients_nonzero = bool(
        not bank_a["numerical_zero"] and not bank_b["numerical_zero"]
    )
    cosine = None
    if gradients_nonzero:
        cosine = float(
            np.dot(gradient_a, gradient_b)
            / (np.linalg.norm(gradient_a) * np.linalg.norm(gradient_b))
        )
    passed = bool(cosine is not None and cosine > COSINE_THRESHOLD)
    return {
        "num_realizations": num_realizations,
        "bank_a": bank_a,
        "bank_b": bank_b,
        "cosine": cosine,
        "sign_table": _sign_table(gradient_a, gradient_b),
        "passed": passed,
    }


def _print_comparison(comparison):
    print(f"## R={comparison['num_realizations']}")
    for label in ("bank_a", "bank_b"):
        bank = comparison[label]
        print(
            f"{label}: stream={bank['stream_id']} "
            f"gradient={bank['gradient']} "
            f"L2={bank['l2_norm']:.16g} "
            f"Linf={bank['linf_norm']:.16g}"
        )
        print(
            f"{label}_mode_difference_counts="
            f"{bank['mode_difference_counts']}"
        )
        print(
            f"{label}_jenkins_difference_counts="
            f"{bank['jenkins_difference_counts']}"
        )
    print(f"cosine={comparison['cosine']}")
    print("coefficient | g_A | g_B | same_sign")
    for row in comparison["sign_table"]:
        print(
            f"{row['coefficient']} | {row['g_A']:.16g} | "
            f"{row['g_B']:.16g} | {row['same_sign']}"
        )


def main() -> int:
    forcing = gate_a_forcing()
    shared_coefficients = np.zeros(
        NUM_FOURIER_COEFFICIENTS, dtype=np.float64
    )
    r4 = _run_comparison(4, forcing, shared_coefficients)
    final = r4
    r8 = None
    if not r4["passed"]:
        r8 = _run_comparison(8, forcing, shared_coefficients)
        final = r8

    passed = final["passed"]
    if passed:
        reason = (
            f"Independent-bank gradient cosine {final['cosine']:.6g} "
            f"exceeds {COSINE_THRESHOLD}."
        )
    elif final["cosine"] is None:
        reason = "At least one independent-bank gradient is numerical-zero."
    else:
        reason = (
            f"Independent-bank gradient cosine {final['cosine']:.6g} "
            f"does not exceed {COSINE_THRESHOLD}."
        )

    results = {
        "configuration": {
            "forcing_seeds": GATE_A_FORCING_SEEDS.tolist(),
            "markov_base_seed": MARKOV_BASE_SEED,
            "markov_iteration": MARKOV_ITERATION,
            "bank_a_stream": BANK_A_STREAM,
            "bank_b_stream": BANK_B_STREAM,
            "num_steps": NUM_STEPS,
            "damping": DAMPING,
            "preload_low": PRELOAD_LOW,
            "preload_high": PRELOAD_HIGH,
            "lambda_0": LAMBDA_0,
            "beta": BETA,
            "fd_epsilon": FD_EPSILON,
            "cosine_threshold": COSINE_THRESHOLD,
        },
        "r4": r4,
        "gate_b": {
            "result": "PASS" if passed else "FAIL",
            "used_realizations": final["num_realizations"],
            "reason": reason,
        },
    }
    if r8 is not None:
        results["r8"] = r8

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")

    print("M1 Direct AD norm = 0 (not recomputed)")
    _print_comparison(r4)
    if r8 is not None:
        _print_comparison(r8)
    print("## Gate B")
    print(f"Gate B: {results['gate_b']['result']}")
    print(f"reason={reason}")
    print(OUTPUT_PATH.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
