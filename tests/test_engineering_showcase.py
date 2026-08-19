import numpy as np

from scripts.run_stage_h4 import H4_TRAINING_SEEDS
from stochastic_stick_slip.engineering_showcase import (
    FIRST_FREQUENCY_RATIO,
    FOURIER_BASIS,
    SECOND_FREQUENCY_RATIO,
    SYSTEM,
    evaluate_controlled_batch,
    preload_history,
)


def test_engineering_showcase_binding() -> None:
    assert SYSTEM.num_free_dofs == 320
    assert np.allclose(SYSTEM.contact_coordinates[:, 0], [0.6875, 0.9375])
    assert (FIRST_FREQUENCY_RATIO, SECOND_FREQUENCY_RATIO) == (1.0, 1.35)

    times = np.asarray(SYSTEM.times)
    expected = np.column_stack(
        (
            np.ones_like(times),
            np.cos(SYSTEM.omega_1 * times),
            np.sin(SYSTEM.omega_1 * times),
            np.cos(1.35 * SYSTEM.omega_1 * times),
            np.sin(1.35 * SYSTEM.omega_1 * times),
        )
    )
    assert np.allclose(np.asarray(FOURIER_BASIS), expected)

    coefficients = np.zeros((8, 5), dtype=np.float64)
    preload = np.asarray(preload_history(0.04, coefficients))
    assert np.array_equal(preload, np.full_like(preload, 0.04))
    result = evaluate_controlled_batch(
        np.array([0.10, 0.04]), coefficients, H4_TRAINING_SEEDS[:8]
    )
    assert np.all(np.isfinite(np.asarray(result.losses)))
    assert np.all(np.asarray(result.stick_to_slip) > 0)
    assert np.all(np.asarray(result.slip_to_stick) > 0)
