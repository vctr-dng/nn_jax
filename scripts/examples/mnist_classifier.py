# %%
import logging
import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm, trange
from utils.mnist import load

from nn_jax import Sequential
from nn_jax.activation import ReLU, Softmax
from nn_jax.layer import Conv2D, Dense
from nn_jax.loss import CategoricalCrossEntropy
from nn_jax.optimizer import StochasticGradientDescent
from nn_jax.pooling import GlobalAvgPool2D, MaxPool2D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# %%
(X_train, Y_train), (X_test, Y_test) = load(n_train=6000, n_test=1000)

fig, axes = plt.subplots(2, 5, figsize=(9, 4))
for ax, image, label in zip(axes.flat, X_train, Y_train.argmax(axis=-1)):
    ax.imshow(image[..., 0], cmap="gray")
    ax.set_title(int(label))
    ax.axis("off")
fig.suptitle("MNIST")
# plt.show()

# %%
conv1_key, conv2_key, dense_key = jax.random.split(jax.random.key(0), 3)

network = Sequential(
    [
        Conv2D(1, 8, (3, 3), key=conv1_key),
        ReLU(),
        MaxPool2D((26, 26, 8), (2, 2), stride=2),
        Conv2D(8, 32, (3, 3), key=conv2_key),
        ReLU(),
        MaxPool2D((11, 11, 32), (2, 2), stride=2),
        # Flatten((5, 5, 16)),
        GlobalAvgPool2D((5, 5, 32)),
        Dense(32, 10, dense_key),
        Softmax(),
    ]
)

loss = CategoricalCrossEntropy()
optimizer = StochasticGradientDescent(learning_rate=0.1)

n_params = 0
for layer in network.modules:
    if getattr(layer, "has_params", False):
        n_params += layer.weights.size + layer.bias.size
logger.info(f"Number of parameters: {n_params}")
logger.info(f"Number of samples: {X_train.shape[0]}")

# %%


def train(
    model: Sequential,
    loss: CategoricalCrossEntropy,
    optimizer: StochasticGradientDescent,
    epochs: int,
    x_train: jax.Array,
    y_train: jax.Array,
    x_test: jax.Array,
    y_test: jax.Array,
    batch_size: int = 64,
) -> tuple[Sequential, list[float], list[float], list[float]]:
    train_errors: list[float] = []
    test_errors: list[float] = []
    test_accuracies: list[float] = []

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

    epoch_bar = trange(epochs, desc="Epoch")
    for i in epoch_bar:
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

        error = 0.0
        n_correct = 0
        n_test_batches = 0
        for start in tqdm(
            range(0, len(x_test), batch_size), desc="Testing", leave=False
        ):
            x_batch = x_test[start : start + batch_size]
            y_batch = y_test[start : start + batch_size]
            predictions = predict(model, x_batch)
            error += float(loss(predictions, y_batch))
            n_correct += int(
                (predictions.argmax(axis=-1) == y_batch.argmax(axis=-1)).sum()
            )
            n_test_batches += 1

        test_errors.append(error / n_test_batches)
        test_accuracies.append(n_correct / len(x_test))
        epoch_bar.set_postfix(
            train_loss=f"{train_errors[-1]:.4f}",
            test_loss=f"{test_errors[-1]:.4f}",
            test_accuracy=f"{test_accuracies[-1]:.2%}",
        )

    return model, train_errors, test_errors, test_accuracies


# %%

network, train_errors, test_errors, test_accuracies = train(
    network,
    loss,
    optimizer,
    epochs=5,
    x_train=X_train,
    y_train=Y_train,
    x_test=X_test,
    y_test=Y_test,
    batch_size=64,
)

# %%

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].plot(train_errors, label="Train")
axes[0].plot(test_errors, label="Test")
axes[0].set_title("Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Categorical cross entropy")
axes[0].legend()

axes[1].plot(test_accuracies)
axes[1].set_title("Test accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
# plt.show()

# %%
predict = jax.jit(jax.vmap(lambda model, x: model.forward(x), in_axes=(None, 0)))
Y_pred = predict(network, X_test)

# %%
true_labels = Y_test.argmax(axis=-1)
predicted_labels = Y_pred.argmax(axis=-1)
num_classes = Y_test.shape[-1]
confusion_matrix = (
    jnp.zeros((num_classes, num_classes), dtype=jnp.int32)
    .at[true_labels, predicted_labels]
    .add(1)
)

true_positives = jnp.diag(confusion_matrix)
predicted_per_class = confusion_matrix.sum(axis=0)
actual_per_class = confusion_matrix.sum(axis=1)
precision = jnp.where(
    predicted_per_class > 0, true_positives / predicted_per_class, 0.0
)
recall = jnp.where(actual_per_class > 0, true_positives / actual_per_class, 0.0)
f1_score = jnp.where(
    precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
)

logger.info(
    msg=f"Test metrics:\naccuracy: {100 * float(true_positives.sum() / confusion_matrix.sum()):.2f}% \n\
macro precision: {100 * float(precision.mean()):.2f}% \n\
macro recall: {100 * float(recall.mean()):.2f}% \n\
macro F1: {100 * float(f1_score.mean()):.2f}%",
)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(
    confusion_matrix,
    annot=True,
    fmt="d",
    cmap="magma_r",
    cbar=False,
    square=True,
    xticklabels=range(num_classes),
    yticklabels=range(num_classes),
    ax=axes[0],
)
axes[0].set_title("Confusion matrix")
axes[0].set_xlabel("Predicted digit")
axes[0].set_ylabel("True digit")

normalized_confusion_matrix = jnp.where(
    actual_per_class[:, None] > 0,
    confusion_matrix / actual_per_class[:, None],
    0.0,
)
sns.heatmap(
    normalized_confusion_matrix,
    annot=True,
    fmt=".2f",
    cmap="magma_r",
    vmin=0,
    vmax=1,
    cbar=False,
    square=True,
    xticklabels=range(num_classes),
    yticklabels=range(num_classes),
    ax=axes[1],
)
axes[1].set_title("Normalized confusion matrix")
axes[1].set_xlabel("Predicted digit")
axes[1].set_ylabel("True digit")
fig.suptitle("Test-set classification metrics")
fig.tight_layout()

metrics = {
    "Precision": precision,
    "Recall": recall,
    "F1 score": f1_score,
}
fig, ax = plt.subplots(figsize=(10, 4))
for name, values in metrics.items():
    ax.plot(range(num_classes), values, marker="o", label=name)
ax.set_title("Per-class metrics")
ax.set_xlabel("Digit")
ax.set_ylabel("Score")
ax.set_xticks(range(num_classes))
ax.set_ylim(0, 1)
ax.legend()

sample_indices = jax.random.choice(jax.random.key(1), len(X_test), (5,), replace=False)

fig, axes = plt.subplots(1, 5, figsize=(9, 2))
for ax, index in zip(axes, sample_indices):
    ax.imshow(X_test[index, ..., 0], cmap="gray")
    ax.set_title(f"pred: {int(Y_pred[index].argmax())}", fontsize=10)
    ax.axis("off")
fig.suptitle("Sample predictions")
# plt.show()

# %%
# single sample through the network, keeping every layer's output so the image transformation can be visualized stage by stage
sample_index = int(jax.random.choice(jax.random.key(2), len(X_test), ()))
sample_x = X_test[sample_index]
sample_label = int(Y_test[sample_index].argmax())

activations: list[tuple[object, str, jax.Array]] = [(None, "Input", sample_x)]
current = sample_x
for i, layer in enumerate(network.modules, start=1):
    current = layer.forward(current)
    activations.append((layer, f"{i}. {type(layer).__name__}", current))

# %%
spatial_stages = [(name, output) for _, name, output in activations if output.ndim == 3]
max_channels = max(output.shape[-1] for _, output in spatial_stages)

fig, axes = plt.subplots(
    len(spatial_stages),
    max_channels,
    figsize=(1.4 * max_channels, 1.4 * len(spatial_stages)),
    squeeze=False,
)
for row, (name, output) in zip(axes, spatial_stages):
    n_channels = output.shape[-1]
    for col, ax in enumerate(row):
        if col < n_channels:
            ax.imshow(output[..., col], cmap="magma_r")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis("off")
    row[0].set_ylabel(name, fontsize=8, rotation=0, ha="right", va="center")
fig.suptitle(f"Feature maps for sample #{sample_index} (true label {sample_label})")
# plt.show()

# %%
# flattened feature vector, the raw (pre-softmax) class scores, and the per-class probabilities the network actually predicts with
# flatten_output = next(o for l, _, o in activations if isinstance(l, Flatten))
flatten_output = next(o for l, _, o in activations if isinstance(l, GlobalAvgPool2D))
dense_output = next(o for l, _, o in activations if isinstance(l, Dense))
probabilities = activations[-1][2]
predicted_label = int(probabilities.argmax())

fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

side = math.isqrt(flatten_output.shape[0])
axes[0].imshow(flatten_output[: side * side].reshape(side, side), cmap="viridis")
axes[0].set_title("Flatten output")
axes[0].axis("off")

axes[1].bar(range(10), dense_output)
axes[1].set_title("Dense output (logits)")
axes[1].set_xlabel("Class")
axes[1].set_xticks(range(10))

colors = ["tab:orange" if i == sample_label else "tab:blue" for i in range(10)]
axes[2].bar(range(10), probabilities, color=colors)
axes[2].set_title("Predicted probability by class")
axes[2].set_xlabel("Class")
axes[2].set_ylabel("Probability")
axes[2].set_xticks(range(10))
axes[2].set_ylim(0, 1)
for i, p in enumerate(probabilities):
    axes[2].text(i, float(p) + 0.02, f"{float(p):.2f}", ha="center", fontsize=7)

fig.suptitle(
    f"Sample #{sample_index} - predicted: {predicted_label}, true: {sample_label} "
    "(true label's bar in orange)"
)
plt.show()
