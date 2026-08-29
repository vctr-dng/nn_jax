# %%
import logging

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

from nn_jax import Sequential, Module
from nn_jax.layer import Dense
from nn_jax.activation import ReLU
from nn_jax.loss import Loss, MeanSquareError
from nn_jax.optimizer import Optimizer, StochasticGradientDescent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#%%
rnd_key = jax.random.PRNGKey(0)

X = jnp.linspace(-20, 20, 1000)
true_Y = X**2
noisy_Y = true_Y + jax.random.normal(rnd_key, X.shape) * 4

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.plot(X, true_Y, linestyle="--", label="True")
ax.plot(X, noisy_Y, label="Noisy")
ax.set_title("$x^2$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend()
# plt.show()

# %%

split_percentage = 0.8

train_indices = jax.random.choice(rnd_key, jnp.arange(len(X)), (int(len(X) * split_percentage),), replace=False)
test_indices = jnp.setdiff1d(jnp.arange(len(X)), train_indices)

X_train = X[train_indices]
Y_train = noisy_Y[train_indices]
X_test = X[test_indices]
Y_test = noisy_Y[test_indices]

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(X_train, Y_train, label="Train", s=5)
ax.scatter(X_test, Y_test, label="Test", s=5)
ax.set_title("$x^2$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend()
# plt.show()

# %%

X_mean, X_std = X_train.mean(), X_train.std()
Y_mean, Y_std = Y_train.mean(), Y_train.std()

X_train_norm = (X_train - X_mean) / X_std
Y_train_norm = (Y_train - Y_mean) / Y_std
X_test_norm = (X_test - X_mean) / X_std
Y_test_norm = (Y_test - Y_mean) / Y_std

# %%

network = Sequential([
    Dense(1, 5),
    ReLU(),
    Dense(5, 5),
    ReLU(),
    Dense(5, 5),
    ReLU(),
    Dense(5, 1),
    ]
)

loss = MeanSquareError()
optimizer = StochasticGradientDescent(network, learning_rate=0.01)

# n_parameters = 
# print(f"Number of parameters: {n_parameters}")

# if X_train.shape[0] < n_parameters:
#     logger.warning(f"Number of samples {X_train.shape[0]} is too low compared to number of parameters {n_parameters}, risk of overfitting")

# %%

def train(
    module: Module,
    loss: Loss,
    optimizer: Optimizer,
    epochs: int,
    x_train: jax.Array,
    y_train: jax.Array,
    x_test: jax.Array | None = None,
    y_test: jax.Array | None = None,
    batch_size: int = 32,
) -> list[float]:
    train_errors: list[float] = []
    test_errors: list[float] = []
    for i in trange(epochs, desc="Epoch"):
        error = 0
        n_batches = 0
        for start in tqdm(range(0, len(x_train), batch_size), desc="Training", leave=False):
            x_batch = x_train[start : start + batch_size].reshape(1, -1)
            y_batch = y_train[start : start + batch_size].reshape(1, -1)
            y_pred = module.forward(x_batch)
            error += loss(y_pred, y_batch)
            loss_derivative = loss.derivative(y_pred, y_batch)
            module.backward(loss_derivative)
            optimizer.step()
            optimizer.zero_grad()
            n_batches += 1

        train_errors.append(error / n_batches)
        # logger.info(f"{i + 1}/{epochs} - Training Error: {train_errors[-1]:.4f}")
        if x_test is not None and y_test is not None:
            error = 0
            n_test_batches = 0
            for start in tqdm(range(0, len(x_test), batch_size), desc="Testing", leave=False):
                x_batch = x_test[start : start + batch_size].reshape(1, -1)
                y_batch = y_test[start : start + batch_size].reshape(1, -1)
                y_pred = module.forward(x_batch)
                error += loss(y_pred, y_batch)
                n_test_batches += 1
            test_errors.append(error / n_test_batches)
            # logger.info(f"{i + 1}/{epochs} - Test Error: {test_errors[-1]:.4f}")

    return train_errors, test_errors

#%%

train_errors, test_errors = train(network, loss, optimizer, 200, X_train_norm, Y_train_norm, X_test_norm, Y_test_norm)

#%%

Y_pred_norm = jnp.array([network.forward(x) for x in X_test_norm]).reshape(-1)
Y_pred = Y_pred_norm * Y_std + Y_mean

fig, ax = plt.subplots(figsize=(4.5, 4.5), ncols=2)
ax[0].plot(train_errors, label="Train")
ax[0].plot(test_errors, label="Test")
ax[0].set_title("$x^2$")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Error")
ax[0].legend()
ax[1].scatter(X_test, Y_test, label="True", s=5)
ax[1].scatter(X_test, Y_pred, label="Predicted", s=5)
ax[1].set_title("$x^2$")
ax[1].set_xlabel("$x$")
ax[1].set_ylabel("$y$")
ax[1].legend()
plt.show()
# %%
