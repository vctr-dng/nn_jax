import jax
import jax.numpy as jnp
import pytest

from nn_jax import Sequential
from nn_jax.activation import ReLU
from nn_jax.layer.conv2d import Conv2D
from nn_jax.layer.dense import Dense
from nn_jax.layer.flatten import Flatten
from nn_jax.pooling.max_pool2d import MaxPool2D


class TestConv2D:
    in_channels = 1
    out_channels = 2
    kernel_size = (3, 3)

    def test_initialization(self):
        conv = Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            key=jax.random.key(0),
        )

        assert conv.in_size == self.in_channels
        assert conv.out_size == self.out_channels
        assert conv.weights.shape == (
            *self.kernel_size,
            self.in_channels,
            self.out_channels,
        )
        assert conv.bias.shape == (self.out_channels,)

    def test_initialization_requires_exactly_one_weight_source(self):
        weights = jnp.zeros((*self.kernel_size, self.in_channels, self.out_channels))

        with pytest.raises(ValueError, match="Key is required"):
            Conv2D(self.in_channels, self.out_channels, self.kernel_size)

        with pytest.raises(ValueError, match="cannot be provided together"):
            Conv2D(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                key=jax.random.key(0),
                weights=weights,
            )

    def test_initialization_validates_parameter_shapes(self):
        with pytest.raises(ValueError, match="Weights shape"):
            Conv2D(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                weights=jnp.zeros((1, 1, 1, 1)),
            )

        with pytest.raises(ValueError, match="Bias shape"):
            Conv2D(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                weights=jnp.zeros(
                    (*self.kernel_size, self.in_channels, self.out_channels)
                ),
                bias=jnp.zeros((1,)),
            )

    def test_forward_shape(self):
        conv = Conv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=2,
            padding=1,
            key=jax.random.key(0),
        )

        output = conv.forward(jnp.ones((5, 5, self.in_channels)))

        assert output.shape == (3, 3, self.out_channels)

    def test_forward_computation(self):
        conv = Conv2D.from_parameters(
            in_channels=1,
            out_channels=1,
            kernel_size=(3, 3),
            weights=jnp.ones((3, 3, 1, 1)),
            bias=jnp.array([0.5]),
        )
        inputs = jnp.arange(1, 17, dtype=jnp.float32).reshape(4, 4, 1)

        expected_output = jnp.array([[[54.5], [63.5]], [[90.5], [99.5]]])

        assert conv.forward(inputs) == pytest.approx(expected_output)

    def test_forward_matches_lax_for_multichannel_strided_convolution(self):
        weights = jnp.arange(36, dtype=jnp.float32).reshape(2, 3, 2, 3) / 10
        bias = jnp.array([0.5, -1.0, 1.5], dtype=jnp.float32)
        conv = Conv2D.from_parameters(
            in_channels=2,
            out_channels=3,
            kernel_size=(2, 3),
            weights=weights,
            bias=bias,
            stride=2,
            padding=1,
        )
        inputs = jnp.arange(60, dtype=jnp.float32).reshape(5, 6, 2) / 10

        expected_output = (
            jax.lax.conv_general_dilated(
                inputs[jnp.newaxis, ...],
                weights,
                window_strides=(2, 2),
                padding=((1, 1), (1, 1)),
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
                precision=jax.lax.Precision.HIGHEST,
            )[0]
            + bias
        )

        assert conv.forward(inputs) == pytest.approx(
            expected_output, rel=2e-4, abs=1e-3
        )

    def test_pytree_round_trip_preserves_parameters_and_metadata(self):
        conv = Conv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            stride=2,
            padding=1,
            key=jax.random.key(0),
        )

        round_tripped = jax.tree.map(lambda value: value, conv)

        assert round_tripped.in_size == conv.in_size
        assert round_tripped.out_size == conv.out_size
        assert round_tripped.kernel_size == conv.kernel_size
        assert round_tripped.stride == conv.stride
        assert round_tripped.padding == conv.padding
        assert round_tripped.weights == pytest.approx(conv.weights)
        assert round_tripped.bias == pytest.approx(conv.bias)

    def test_grad_matches_numerical_gradient(self):
        conv = Conv2D.from_parameters(
            in_channels=1,
            out_channels=1,
            kernel_size=(2, 2),
            weights=jnp.array([[[[0.1]], [[0.2]]], [[[0.3]], [[0.4]]]]),
            bias=jnp.array([0.5]),
        )
        inputs = jnp.arange(1, 10, dtype=jnp.float32).reshape(3, 3, 1)
        output_coefficients = jnp.arange(1, 5, dtype=jnp.float32).reshape(2, 2, 1)

        def weighted_output(layer, x):
            return jnp.sum(layer.forward(x) * output_coefficients)

        def central_difference(function, value, index, epsilon=1e-3):
            positive = value.at[index].add(epsilon)
            negative = value.at[index].add(-epsilon)
            return (function(positive) - function(negative)) / (2 * epsilon)

        conv_grad, input_grad = jax.grad(weighted_output, argnums=(0, 1))(conv, inputs)

        numerical_weight_grad = central_difference(
            lambda weights: weighted_output(
                Conv2D.from_parameters(1, 1, (2, 2), weights, conv.bias), inputs
            ),
            conv.weights,
            (1, 0, 0, 0),
        )
        numerical_bias_grad = central_difference(
            lambda bias: weighted_output(
                Conv2D.from_parameters(1, 1, (2, 2), conv.weights, bias), inputs
            ),
            conv.bias,
            (0,),
        )
        numerical_input_grad = central_difference(
            lambda x: weighted_output(conv, x), inputs, (1, 1, 0)
        )

        assert conv_grad.weights[1, 0, 0, 0] == pytest.approx(
            numerical_weight_grad, rel=3e-3, abs=5e-3
        )
        assert conv_grad.bias[0] == pytest.approx(
            numerical_bias_grad, rel=3e-3, abs=5e-3
        )
        assert input_grad[1, 1, 0] == pytest.approx(
            numerical_input_grad, rel=3e-3, abs=5e-3
        )

    def test_cnn_stack_has_expected_output_and_gradient_shapes(self):
        conv_key, dense_key = jax.random.split(jax.random.key(0))
        network = Sequential(
            [
                Conv2D(2, 3, (3, 3), key=conv_key, padding=1),
                ReLU(),
                MaxPool2D((6, 6, 3), (2, 2), stride=2),
                Flatten((3, 3, 3)),
                Dense(27, 4, key=dense_key),
            ]
        )
        inputs = jnp.arange(72, dtype=jnp.float32).reshape(6, 6, 2) / 72

        output = network.forward(inputs)
        grads = jax.grad(lambda model: jnp.sum(model.forward(inputs)))(network)

        assert output.shape == (4,)
        assert grads.modules[0].weights.shape == network.modules[0].weights.shape
        assert grads.modules[0].bias.shape == network.modules[0].bias.shape
        assert grads.modules[-1].weights.shape == network.modules[-1].weights.shape
        assert grads.modules[-1].bias.shape == network.modules[-1].bias.shape
        assert bool(jnp.all(jnp.isfinite(grads.modules[0].weights)))
        assert bool(jnp.all(jnp.isfinite(grads.modules[-1].weights)))

    def test_jit_and_vmap_forward_match_eager(self):
        conv = Conv2D(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            key=jax.random.key(0),
        )
        inputs = jnp.arange(32, dtype=jnp.float32).reshape(2, 4, 4, 1)

        compiled_forward = jax.jit(jax.vmap(conv.forward))

        assert compiled_forward(inputs) == pytest.approx(jax.vmap(conv.forward)(inputs))
