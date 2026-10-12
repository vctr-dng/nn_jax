import jax.numpy as jnp
from jax import Array


def extract_patches(x: Array, kernel_size: tuple[int, int], stride: int) -> Array:
    H_in, W_in, _ = x.shape
    H_ker, W_ker = kernel_size
    H_out = output_spatial_size(H_in, H_ker, stride, 0)
    W_out = output_spatial_size(W_in, W_ker, stride, 0)

    base_rows = jnp.arange(H_out) * stride
    base_cols = jnp.arange(W_out) * stride
    kernel_rows = jnp.arange(H_ker)
    kernel_cols = jnp.arange(W_ker)

    rows = base_rows[:, None, None, None] + kernel_rows[None, None, :, None]
    cols = base_cols[None, :, None, None] + kernel_cols[None, None, None, :]
    patches = x[rows, cols]

    return patches


def output_spatial_size(
    in_size: int, kernel_size: int, stride: int, padding: int
) -> int:
    if in_size <= 0:
        raise ValueError("Input size must be positive")
    if kernel_size <= 0:
        raise ValueError("Kernel size must be positive")
    if stride <= 0:
        raise ValueError("Stride must be positive")
    if padding < 0:
        raise ValueError("Padding must be non-negative")
    if kernel_size > in_size + 2 * padding:
        raise ValueError("Kernel size cannot exceed padded input size")

    return (in_size + 2 * padding - kernel_size) // stride + 1
