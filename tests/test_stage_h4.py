import numpy as np

from scripts.run_stage_h4 import H4_TEST_SEEDS, H4_TRAINING_SEEDS
from stochastic_stick_slip.model import TRAINING_SEEDS


def test_h4_training_seeds_are_fixed_and_complete() -> None:
    expected = np.concatenate(
        (TRAINING_SEEDS, np.arange(201, 225, dtype=np.int64))
    )
    assert len(H4_TRAINING_SEEDS) == 32
    assert len(np.unique(H4_TRAINING_SEEDS)) == 32
    assert np.array_equal(H4_TRAINING_SEEDS, expected)


def test_h4_test_seeds_are_disjoint() -> None:
    assert np.array_equal(
        H4_TEST_SEEDS, np.arange(1001, 1065, dtype=np.int64)
    )
    assert set(H4_TEST_SEEDS).isdisjoint(H4_TRAINING_SEEDS)
