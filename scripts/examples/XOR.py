# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nn_jax import Module, Sequential
from nn_jax.activation import Tanh
from nn_jax.layer import Dense
from nn_jax.loss import Loss, MeanSquareError
from nn_jax.optimizer import Optimizer, StochasticGradientDescent

# %%
X = jnp.reshape(jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (4, 2, 1))
Y_true = jnp.reshape(jnp.array([[0], [1], [1], [0]]), (4, 1, 1))

fig, ax = plt.subplots(figsize=(4.5, 4))
sc = ax.scatter(
    X[:, 0, 0],
    X[:, 1, 0],
    c=Y_true[:, 0, 0],
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
network = Sequential([Dense(2, 3), Tanh(), Dense(3, 1), Tanh()])


def train(
    module: Module,
    x_train: jax.Array,
    y_train: jax.Array,
    loss: Loss,
    optimizer: Optimizer,
    epochs: int,
) -> list[float]:
    errors: list[float] = []

    for i in range(epochs):
        error = 0
        for x, y_true in zip(x_train, y_train):
            y_pred = module.forward(x)
            error += loss(y_pred, y_true)
            loss_derivative = loss.derivative(y_pred, y_true)
            module.backward(loss_derivative)
            optimizer.step()
            optimizer.zero_grad()

        errors.append(error / len(x_train))
        if i % 100 == 0:
            print(f"{i + 1}/{epochs} - Error: {errors[-1]:.4f}")

    return errors


optimizer = StochasticGradientDescent(network, learning_rate=0.05)

errors = train(network, X, Y_true, MeanSquareError(), optimizer, 1000)

plt.plot(errors)
plt.show()

# %%

Y_pred: list[jax.Array] = []

for x in X:
    y_pred = network.forward(x)
    Y_pred.append(y_pred)

Y_pred: jax.Array = jnp.array(Y_pred)

fig, ax = plt.subplots(figsize=(4.5, 4))
sc = ax.scatter(
    X[:, 0, 0],
    X[:, 1, 0],
    c=Y_pred[:, 0, 0],
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
