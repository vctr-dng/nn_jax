# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nn_jax import Sequential
from nn_jax.activation import Tanh
from nn_jax.layer import Dense
from nn_jax.loss import MeanSquareError
from nn_jax.optimizer import StochasticGradientDescent

# %%
X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
Y_true = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)

fig, ax = plt.subplots(figsize=(4.5, 4))
sc = ax.scatter(
    X[:, 0],
    X[:, 1],
    c=Y_true[:, 0],
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
)
ax.set_title("XOR")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
cbar = fig.colorbar(sc, ax=ax, label="$y$")
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.show()

# %%
layer_keys = jax.random.split(jax.random.key(0), 2)
network = Sequential(
    [Dense(2, 3, layer_keys[0]), Tanh(), Dense(3, 1, layer_keys[1]), Tanh()]
)


def train(
    model: Sequential,
    x_train: jax.Array,
    y_train: jax.Array,
    loss: MeanSquareError,
    optimizer: StochasticGradientDescent,
    epochs: int,
) -> tuple[Sequential, list[float]]:
    errors: list[float] = []

    def loss_fn(current_model: Sequential, x_batch: jax.Array, y_batch: jax.Array):
        predictions = jax.vmap(current_model.forward)(x_batch)
        return loss(predictions, y_batch)

    @jax.jit
    def train_step(current_model: Sequential, x_batch: jax.Array, y_batch: jax.Array):
        error, grads = jax.value_and_grad(loss_fn)(current_model, x_batch, y_batch)
        return optimizer(current_model, grads), error

    for i in range(epochs):
        model, error = train_step(model, x_train, y_train)

        errors.append(float(error))
        if i % 100 == 0:
            print(f"{i + 1}/{epochs} - Error: {errors[-1]:.4f}")

    return model, errors


optimizer = StochasticGradientDescent(learning_rate=0.05)

network, errors = train(network, X, Y_true, MeanSquareError(), optimizer, 2000)

plt.plot(errors)
plt.show()

# %%

predict = jax.jit(jax.vmap(lambda model, x: model.forward(x), in_axes=(None, 0)))
Y_pred = predict(network, X)

fig, ax = plt.subplots(figsize=(4.5, 4))
sc = ax.scatter(
    X[:, 0],
    X[:, 1],
    c=Y_pred[:, 0],
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
)
ax.set_title("XOR")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
cbar = fig.colorbar(sc, ax=ax, label="$y$")
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
plt.show()
