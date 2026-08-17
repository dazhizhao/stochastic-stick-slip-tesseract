"""Check the local Python scientific-computing environment."""

import platform
import sys
from importlib.metadata import version

import jax
import jax.numpy as jnp
import jax_fem
import torch
import tesseract_core
from tesseract_core import Tesseract
import tesseract_jax
import tesseract_torch


print(f"Python version: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"Machine architecture: {platform.machine()}")

print(f"JAX version: {version('jax')}")
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")
x = jnp.array([1.0, 2.0])
print(f"JAX sum(x**2): {jnp.sum(x**2)}")
print(f"JAX grad(x**2) at 2: {jax.grad(lambda value: value**2)(2.0)}")

print(f"JAX-FEM import: OK ({version('jax-fem')})")

print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch MPS available: {torch.backends.mps.is_available()}")
print(f"PyTorch MPS built: {torch.backends.mps.is_built()}")
print(f"PyTorch CPU tensor sum: {torch.tensor([1.0, 2.0]).sum().item()}")

print(f"Tesseract Core import: OK ({version('tesseract-core')})")
print(f"Tesseract JAX import: OK ({version('tesseract-jax')})")
print(f"Tesseract Torch import: OK ({version('tesseract-torch')})")
