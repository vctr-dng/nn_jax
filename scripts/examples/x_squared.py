# %%
import logging

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm, trange

from nn_jax import Sequential
from nn_jax.activation import ReLU
from nn_jax.layer import Dense
from nn_jax.loss import MeanSquareError
from nn_jax.optimizer import StochasticGradientDescent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %%
noise_key, split_key, *layer_keys = jax.random.split(jax.random.key(0), 6)

X = jnp.linspace(-50, 50, 5000)
true_Y = X**2
noisy_Y = true_Y + jax.random.normal(noise_key, X.shape) * 4

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

train_indices = jax.random.choice(
    split_key, jnp.arange(len(X)), (int(len(X) * split_percentage),), replace=False
)
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

X_train_norm = ((X_train - X_mean) / X_std)[:, None]
Y_train_norm = ((Y_train - Y_mean) / Y_std)[:, None]
X_test_norm = ((X_test - X_mean) / X_std)[:, None]
Y_test_norm = ((Y_test - Y_mean) / Y_std)[:, None]

# %%

network = Sequential(
    [
        Dense(1, 10, layer_keys[0]),
        ReLU(),
        Dense(10, 10, layer_keys[1]),
        ReLU(),
        Dense(10, 10, layer_keys[2]),
        ReLU(),
        Dense(10, 1, layer_keys[3]),
    ]
)

loss = MeanSquareError()
optimizer = StochasticGradientDescent(learning_rate=0.01)

n_params = 0
for layer in network.modules:
    if isinstance(layer, Dense):
        n_params += layer.weights.size + layer.bias.size
logger.info(f"Number of parameters: {n_params}")
logger.info(f"Number of samples: {X_train.shape[0]}")
# %%


def train(
    model: Sequential,
    loss: MeanSquareError,
    optimizer: StochasticGradientDescent,
    epochs: int,
    x_train: jax.Array,
    y_train: jax.Array,
    x_test: jax.Array | None = None,
    y_test: jax.Array | None = None,
    batch_size: int = 32,
) -> tuple[Sequential, list[float], list[float]]:
    train_errors: list[float] = []
    test_errors: list[float] = []

    def loss_fn(current_model: Sequential, x_batch: jax.Array, y_batch: jax.Array):
        predictions = jax.vmap(current_model.forward)(x_batch)
        return loss(predictions, y_batch)

    @jax.jit
    def train_step(current_model: Sequential, x_batch: jax.Array, y_batch: jax.Array):
        error, grads = jax.value_and_grad(loss_fn)(current_model, x_batch, y_batch)
        return optimizer(current_model, grads), error

    predict = jax.jit(
        jax.vmap(lambda current_model, x: current_model.forward(x), in_axes=(None, 0))
    )

    for i in trange(epochs, desc="Epoch"):
        error = 0.0
        n_batches = 0
        for start in tqdm(
            range(0, len(x_train), batch_size), desc="Training", leave=False
        ):
            x_batch = x_train[start : start + batch_size]
            y_batch = y_train[start : start + batch_size]
            model, batch_error = train_step(model, x_batch, y_batch)
            error += float(batch_error)
            n_batches += 1

        train_errors.append(error / n_batches)
        if x_test is not None and y_test is not None:
            error = 0.0
            n_test_batches = 0
            for start in tqdm(
                range(0, len(x_test), batch_size), desc="Testing", leave=False
            ):
                x_batch = x_test[start : start + batch_size]
                y_batch = y_test[start : start + batch_size]
                error += float(loss(predict(model, x_batch), y_batch))
                n_test_batches += 1
            test_errors.append(error / n_test_batches)

    return model, train_errors, test_errors


# %%

network, train_errors, test_errors = train(
    network,
    loss,
    optimizer,
    100,
    X_train_norm,
    Y_train_norm,
    X_test_norm,
    Y_test_norm,
    batch_size=64,
)

# %%

predict = jax.jit(jax.vmap(lambda model, x: model.forward(x), in_axes=(None, 0)))
Y_pred_norm = predict(network, X_test_norm).reshape(-1)
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
