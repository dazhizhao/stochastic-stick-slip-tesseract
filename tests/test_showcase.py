import numpy as np

from stochastic_stick_slip.showcase import (
    CONTACT_COLUMNS,
    NUM_ELEMENTS_X,
    NUM_ELEMENTS_Y,
    SYSTEM,
)


def test_showcase_mesh_and_contacts() -> None:
    assert (NUM_ELEMENTS_X, NUM_ELEMENTS_Y) == (32, 4)
    assert len(SYSTEM.cells) == 128
    assert len(SYSTEM.points) == 165
    assert SYSTEM.num_total_dofs == 330
    assert SYSTEM.num_free_dofs == 320
    assert len(SYSTEM.fixed_dofs) == 10
    expected_x = np.asarray(CONTACT_COLUMNS, dtype=np.float64) / 32.0
    assert np.allclose(SYSTEM.contact_coordinates[:, 0], expected_x)
    assert np.array_equal(SYSTEM.contact_coordinates[:, 1], np.zeros(2))
    assert np.array_equal(
        np.flatnonzero(np.asarray(SYSTEM.contacts)[:, 0]),
        [np.flatnonzero(SYSTEM.free_dofs == 2 * SYSTEM.contact_nodes[0] + 1)[0]],
    )
    assert np.array_equal(
        np.flatnonzero(np.asarray(SYSTEM.contacts)[:, 1]),
        [np.flatnonzero(SYSTEM.free_dofs == 2 * SYSTEM.contact_nodes[1] + 1)[0]],
    )
