"""Hard one-contact Jenkins return mapping."""

import jax.numpy as jnp


def select_contact_state(
    stick_displacement,
    slider_displacement,
    tangential_stiffness,
    friction_limit,
):
    """Return trial force, slip flag, and slip direction."""
    trial_force = tangential_stiffness * (
        stick_displacement - slider_displacement
    )
    tolerance = 1e-12 * (1.0 + friction_limit)
    is_slipping = jnp.abs(trial_force) > friction_limit + tolerance
    direction = jnp.where(trial_force >= 0.0, 1.0, -1.0)
    return trial_force, is_slipping, direction
