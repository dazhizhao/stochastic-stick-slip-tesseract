import jax
import jax_fem
import pytest
import torch
import tesseract_jax
import tesseract_torch
from tesseract_core import Tesseract


def test_required_imports() -> None:
    assert jax is not None
    assert jax_fem is not None
    assert torch is not None
    assert Tesseract is not None
    assert tesseract_jax is not None
    assert tesseract_torch is not None


def test_jax_grad() -> None:
    gradient = jax.grad(lambda value: value**2)(2.0)
    assert gradient == pytest.approx(4.0)
