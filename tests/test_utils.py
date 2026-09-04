import jax.numpy as jnp
import pytest

from nn_jax.utils.im2col import extract_patches, output_spatial_size


class TestExtractPatches:
    def test_extracts_non_overlapping_hwc_patches(self):
        inputs = jnp.arange(16, dtype=jnp.float32).reshape(4, 4, 1)

        patches = extract_patches(inputs, kernel_size=(2, 2), stride=2)

        expected_patches = jnp.array(
            [
                [
                    [[[0.0], [1.0]], [[4.0], [5.0]]],
                    [[[2.0], [3.0]], [[6.0], [7.0]]],
                ],
                [
                    [[[8.0], [9.0]], [[12.0], [13.0]]],
                    [[[10.0], [11.0]], [[14.0], [15.0]]],
                ],
            ]
        )

        assert patches.shape == (2, 2, 2, 2, 1)
        assert patches == pytest.approx(expected_patches)

    def test_extracts_overlapping_patches_for_each_channel(self):
        inputs = jnp.arange(18, dtype=jnp.float32).reshape(3, 3, 2)

        patches = extract_patches(inputs, kernel_size=(2, 2), stride=1)

        expected_patches = jnp.array(
            [
                [
                    [[[0.0, 1.0], [2.0, 3.0]], [[6.0, 7.0], [8.0, 9.0]]],
                    [
                        [[2.0, 3.0], [4.0, 5.0]],
                        [[8.0, 9.0], [10.0, 11.0]],
                    ],
                ],
                [
                    [
                        [[6.0, 7.0], [8.0, 9.0]],
                        [[12.0, 13.0], [14.0, 15.0]],
                    ],
                    [
                        [[8.0, 9.0], [10.0, 11.0]],
                        [[14.0, 15.0], [16.0, 17.0]],
                    ],
                ],
            ]
        )

        assert patches.shape == (2, 2, 2, 2, 2)
        assert patches == pytest.approx(expected_patches)


class TestOutputSpatialSize:
    @pytest.mark.parametrize(
        ("in_size", "kernel_size", "stride", "padding", "expected_size"),
        [
            (4, 3, 1, 0, 2),
            (4, 3, 1, 1, 4),
            (5, 2, 2, 0, 2),
            (7, 3, 2, 1, 4),
        ],
    )
    def test_calculates_output_size(
        self, in_size, kernel_size, stride, padding, expected_size
    ):
        assert (
            output_spatial_size(in_size, kernel_size, stride, padding) == expected_size
        )

    @pytest.mark.parametrize(
        ("in_size", "kernel_size", "stride", "padding", "message"),
        [
            (0, 2, 1, 0, "Input size"),
            (4, 0, 1, 0, "Kernel size"),
            (4, 2, 0, 0, "Stride"),
            (4, 2, 1, -1, "Padding"),
            (2, 3, 1, 0, "cannot exceed"),
        ],
    )
    def test_rejects_invalid_geometry(
        self, in_size, kernel_size, stride, padding, message
    ):
        with pytest.raises(ValueError, match=message):
            output_spatial_size(in_size, kernel_size, stride, padding)
