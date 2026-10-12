# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "anywidget>=0.9.21",
#     "huggingface-hub>=1.29.0",
#     "jax[cuda13]>=0.8.0",
#     "marimo>=0.20.4",
#     "matplotlib>=3.10.8",
#     "pillow>=12.3.0",
#     "pyarrow>=25.0.1",
#     "seaborn>=0.13.2",
# ]
# ///
# ruff: noqa: B018, PLR1711

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import math

    import anywidget
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    import traitlets
    from matplotlib.patches import Rectangle
    from utils.mnist import load

    from nn_jax import Sequential
    from nn_jax.activation import ReLU, Softmax
    from nn_jax.layer import Conv2D, Dense
    from nn_jax.loss import CategoricalCrossEntropy
    from nn_jax.optimizer import StochasticGradientDescent
    from nn_jax.pooling import GlobalAvgPool2D, MaxPool2D

    return (
        CategoricalCrossEntropy,
        Conv2D,
        Dense,
        GlobalAvgPool2D,
        MaxPool2D,
        ReLU,
        Rectangle,
        Sequential,
        Softmax,
        StochasticGradientDescent,
        anywidget,
        jax,
        jnp,
        load,
        math,
        mo,
        plt,
        sns,
        traitlets,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # MNIST CNN classifier

    A CNN built from `nn_jax` primitives and trained on MNIST.
    """)
    return


@app.cell
def _(mo):
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(load):
    (X_train, Y_train), (X_test, Y_test) = load(n_train=6000, n_test=1000)
    return X_test, X_train, Y_test, Y_train


@app.cell(hide_code=True)
def _(X_train, Y_train, plt):
    sample_fig, sample_axes = plt.subplots(2, 5, figsize=(9, 4))
    for sample_axis, image, label in zip(
        sample_axes.flat, X_train, Y_train.argmax(axis=-1)
    ):
        sample_axis.imshow(image[..., 0], cmap="gray")
        sample_axis.set_title(int(label))
        sample_axis.axis("off")
    sample_fig.suptitle("MNIST")
    sample_fig
    return


@app.cell
def _(
    CategoricalCrossEntropy,
    Conv2D,
    Dense,
    GlobalAvgPool2D,
    MaxPool2D,
    ReLU,
    Sequential,
    Softmax,
    StochasticGradientDescent,
    jax,
):
    conv1_key, conv2_key, dense_key = jax.random.split(jax.random.key(0), 3)
    network = Sequential(
        [
            Conv2D(1, 8, (3, 3), key=conv1_key),
            ReLU(),
            MaxPool2D((26, 26, 8), (2, 2), stride=2),
            Conv2D(8, 32, (3, 3), key=conv2_key),
            ReLU(),
            MaxPool2D((11, 11, 32), (2, 2), stride=2),
            GlobalAvgPool2D((5, 5, 32)),
            Dense(32, 10, dense_key),
            Softmax(),
        ]
    )
    loss = CategoricalCrossEntropy()
    optimizer = StochasticGradientDescent(learning_rate=0.1)
    return loss, network, optimizer


@app.cell
def _(X_train, mo, network):
    parameter_count = sum(
        layer.weights.size + layer.bias.size
        for layer in network.modules
        if getattr(layer, "has_params", False)
    )
    mo.md(
        f"**Parameters:** {parameter_count:,}  \n**Training samples:** {X_train.shape[0]:,}"
    )
    return


@app.cell
def _(Sequential, jax, mo):
    def train(
        model: Sequential,
        loss,
        optimizer,
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
        def train_step(
            current_model: Sequential, x_batch: jax.Array, y_batch: jax.Array
        ):
            error, grads = jax.value_and_grad(loss_fn)(current_model, x_batch, y_batch)
            return optimizer(current_model, grads), error

        predict = jax.jit(
            jax.vmap(
                lambda current_model, inputs: current_model.forward(inputs),
                in_axes=(None, 0),
            )
        )

        for _ in mo.status.progress_bar(
            range(epochs),
            title="Training CNN",
            completion_title="Training complete",
        ):
            train_error = 0.0
            train_batches = 0
            for start in range(0, len(x_train), batch_size):
                x_batch = x_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]
                model, batch_error = train_step(model, x_batch, y_batch)
                train_error += float(batch_error)
                train_batches += 1
            train_errors.append(train_error / train_batches)

            test_error = 0.0
            correct_predictions = 0
            test_batches = 0
            for start in range(0, len(x_test), batch_size):
                x_batch = x_test[start : start + batch_size]
                y_batch = y_test[start : start + batch_size]
                predictions = predict(model, x_batch)
                test_error += float(loss(predictions, y_batch))
                correct_predictions += int(
                    (predictions.argmax(axis=-1) == y_batch.argmax(axis=-1)).sum()
                )
                test_batches += 1
            test_errors.append(test_error / test_batches)
            test_accuracies.append(correct_predictions / len(x_test))

        return model, train_errors, test_errors, test_accuracies

    return (train,)


@app.cell
def _(X_test, X_train, Y_test, Y_train, loss, network, optimizer, train):
    trained_network, train_errors, test_errors, test_accuracies = train(
        network,
        loss,
        optimizer,
        epochs=35,
        x_train=X_train,
        y_train=Y_train,
        x_test=X_test,
        y_test=Y_test,
        batch_size=64,
    )
    return test_accuracies, test_errors, train_errors, trained_network


@app.cell(hide_code=True)
def _(plt, test_accuracies, test_errors, train_errors):
    history_fig, history_axes = plt.subplots(1, 2, figsize=(9, 4))
    history_axes[0].plot(train_errors, label="Train")
    history_axes[0].plot(test_errors, label="Test")
    history_axes[0].set_title("Loss")
    history_axes[0].set_xlabel("Epoch")
    history_axes[0].set_ylabel("Categorical cross entropy")
    history_axes[0].legend()
    history_axes[1].plot(test_accuracies)
    history_axes[1].set_title("Test accuracy")
    history_axes[1].set_xlabel("Epoch")
    history_axes[1].set_ylabel("Accuracy")
    history_fig
    return


@app.cell
def _(X_test, jax, trained_network):
    predict = jax.jit(
        jax.vmap(
            lambda current_model, inputs: current_model.forward(inputs),
            in_axes=(None, 0),
        )
    )
    Y_pred = predict(trained_network, X_test)
    return (Y_pred,)


@app.cell
def _(Y_pred, Y_test, jnp):
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
    accuracy = true_positives.sum() / confusion_matrix.sum()
    return accuracy, confusion_matrix, f1_score, precision, recall


@app.cell(hide_code=True)
def _(accuracy, f1_score, mo, precision, recall):
    mo.md(f"""
    ## Test metrics

        - Accuracy: {100 * float(accuracy):.2f}%
        - Macro precision: {100 * float(precision.mean()):.2f}%
        - Macro recall: {100 * float(recall.mean()):.2f}%
        - Macro F1: {100 * float(f1_score.mean()):.2f}%
    """)
    return


@app.cell(hide_code=True)
def _(confusion_matrix, plt, sns):
    matrix_fig, matrix_axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="magma_r",
        cbar=False,
        square=True,
        xticklabels=range(confusion_matrix.shape[1]),
        yticklabels=range(confusion_matrix.shape[0]),
        ax=matrix_axes[0],
    )
    matrix_axes[0].set_title("Confusion matrix")
    matrix_axes[0].set_xlabel("Predicted digit")
    matrix_axes[0].set_ylabel("True digit")

    normalization_counts = confusion_matrix.sum(axis=1)
    normalized_confusion_matrix = confusion_matrix / normalization_counts[:, None]
    sns.heatmap(
        normalized_confusion_matrix,
        annot=True,
        fmt=".2f",
        cmap="magma_r",
        vmin=0,
        vmax=1,
        cbar=False,
        square=True,
        xticklabels=range(confusion_matrix.shape[1]),
        yticklabels=range(confusion_matrix.shape[0]),
        ax=matrix_axes[1],
    )
    matrix_axes[1].set_title("Normalized confusion matrix")
    matrix_axes[1].set_xlabel("Predicted digit")
    matrix_axes[1].set_ylabel("True digit")
    matrix_fig.suptitle("Test-set classification metrics")
    matrix_fig.tight_layout()
    matrix_fig
    return


@app.cell(hide_code=True)
def _(f1_score, plt, precision, recall):
    metrics_fig, metrics_axis = plt.subplots(figsize=(10, 4))
    for metric_name, values in {
        "Precision": precision,
        "Recall": recall,
        "F1 score": f1_score,
    }.items():
        metrics_axis.plot(range(len(values)), values, marker="o", label=metric_name)
    metrics_axis.set_title("Per-class metrics")
    metrics_axis.set_xlabel("Digit")
    metrics_axis.set_ylabel("Score")
    metrics_axis.set_xticks(range(len(precision)))
    metrics_axis.set_ylim(0, 1)
    metrics_axis.legend()
    metrics_fig
    return


@app.cell(hide_code=True)
def _(X_test, Y_pred, jax, plt):
    sample_indices = jax.random.choice(
        jax.random.key(1), len(X_test), (5,), replace=False
    )
    predictions_fig, predictions_axes = plt.subplots(1, 5, figsize=(9, 2))
    for prediction_axis, prediction_index in zip(predictions_axes, sample_indices):
        prediction_axis.imshow(X_test[prediction_index, ..., 0], cmap="gray")
        prediction_axis.set_title(
            f"pred: {int(Y_pred[prediction_index].argmax())}", fontsize=10
        )
        prediction_axis.axis("off")
    predictions_fig.suptitle("Sample predictions")
    predictions_fig
    return


@app.cell(hide_code=True)
def _(anywidget, traitlets):
    class DigitCanvas(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          const gridSize = 28;
          const canvasSize = 336;
          const container = document.createElement("div");
          const canvas = document.createElement("canvas");
          const clearButton = document.createElement("button");
          const context = canvas.getContext("2d");
          let drawing = false;
          let pixels = model.get("pixels");

          container.className = "digit-canvas";
          canvas.width = canvasSize;
          canvas.height = canvasSize;
          canvas.setAttribute("aria-label", "Draw a digit");
          clearButton.textContent = "Clear";
          container.append(canvas, clearButton);
          el.appendChild(container);

          const resetPixels = () => Array(gridSize * gridSize).fill(0);
          const readPixels = () => {
            const value = model.get("pixels");
            return Array.isArray(value) && value.length === gridSize * gridSize
              ? [...value]
              : resetPixels();
          };
          const renderPixels = () => {
            const cellSize = canvasSize / gridSize;
            context.fillStyle = "#05070b";
            context.fillRect(0, 0, canvasSize, canvasSize);
            for (let row = 0; row < gridSize; row += 1) {
              for (let column = 0; column < gridSize; column += 1) {
                const value = pixels[row * gridSize + column];
                const shade = Math.round(value * 255);
                context.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
                context.fillRect(
                  column * cellSize,
                  row * cellSize,
                  cellSize + 0.5,
                  cellSize + 0.5,
                );
              }
            }
          };
          const savePixels = () => {
            model.set("pixels", pixels);
            model.save_changes();
          };
          const paint = (event) => {
            const bounds = canvas.getBoundingClientRect();
            const column = Math.floor(
              ((event.clientX - bounds.left) / bounds.width) * gridSize,
            );
            const row = Math.floor(
              ((event.clientY - bounds.top) / bounds.height) * gridSize,
            );
            for (let deltaRow = -2; deltaRow <= 2; deltaRow += 1) {
              for (let deltaColumn = -2; deltaColumn <= 2; deltaColumn += 1) {
                const targetRow = row + deltaRow;
                const targetColumn = column + deltaColumn;
                const distance = Math.hypot(deltaRow, deltaColumn);
                if (
                  targetRow >= 0 && targetRow < gridSize &&
                  targetColumn >= 0 && targetColumn < gridSize &&
                  distance <= 2.5
                ) {
                  const index = targetRow * gridSize + targetColumn;
                  pixels[index] = Math.max(pixels[index], 1 - distance / 2.5);
                }
              }
            }
            renderPixels();
          };
          const endStroke = () => {
            if (drawing) {
              drawing = false;
              savePixels();
            }
          };

          canvas.addEventListener("pointerdown", (event) => {
            drawing = true;
            canvas.setPointerCapture(event.pointerId);
            paint(event);
          });
          canvas.addEventListener("pointermove", (event) => {
            if (drawing) paint(event);
          });
          canvas.addEventListener("pointerup", endStroke);
          canvas.addEventListener("pointercancel", endStroke);
          clearButton.addEventListener("click", () => {
            pixels = resetPixels();
            renderPixels();
            savePixels();
          });
          model.on("change:pixels", () => {
            pixels = readPixels();
            renderPixels();
          });
          pixels = readPixels();
          renderPixels();
        }

        export default { render };
        """
        _css = """
        .digit-canvas {
          display: inline-flex;
          flex-direction: column;
          gap: 0.75rem;
          align-items: flex-start;
          padding: 1rem;
          background: #f4f7fb;
          border: 1px solid #b9c5d5;
          border-radius: 0.75rem;
        }
        .digit-canvas canvas {
          width: min(336px, 100%);
          aspect-ratio: 1;
          image-rendering: pixelated;
          cursor: crosshair;
          border: 1px solid #30445c;
          touch-action: none;
        }
        .digit-canvas button {
          padding: 0.4rem 0.8rem;
          color: #fff;
          background: #244d78;
          border: 0;
          border-radius: 0.35rem;
          cursor: pointer;
        }
        .digit-canvas button:hover { background: #163956; }
        @media (prefers-color-scheme: dark) {
          .digit-canvas { background: #151b24; border-color: #3f4d61; }
          .digit-canvas canvas { border-color: #91a8c2; }
          .digit-canvas button { background: #5b94c7; }
          .digit-canvas button:hover { background: #77ace0; }
        }
        """
        pixels = traitlets.List(
            trait=traitlets.Float(), default_value=[0.0] * (28 * 28)
        ).tag(sync=True)

    return (DigitCanvas,)


@app.cell
def _(DigitCanvas, mo):
    digit_canvas = mo.ui.anywidget(DigitCanvas())
    mo.vstack([mo.md("## Draw a digit"), digit_canvas])
    return (digit_canvas,)


@app.cell
def _(X_test, digit_canvas, is_script_mode, jnp):
    if is_script_mode:
        feature_input = X_test[0]
        feature_source = "MNIST test sample 0"
    else:
        feature_input = jnp.asarray(digit_canvas.value["pixels"], dtype=X_test.dtype)
        feature_input = feature_input.reshape(28, 28, 1)
        feature_source = "your drawing"
    return feature_input, feature_source


@app.cell
def _(GlobalAvgPool2D, feature_input, trained_network):
    activations = [(None, "Input", feature_input)]
    current = feature_input
    for layer_index, layer in enumerate(trained_network.modules, start=1):
        current = layer.forward(current)
        activations.append((layer, f"{layer_index}. {type(layer).__name__}", current))
    pooled_output = next(
        output for layer, _, output in activations if isinstance(layer, GlobalAvgPool2D)
    )
    return activations, pooled_output


@app.cell(hide_code=True, expand_output=True)
def _(Rectangle, activations, feature_source, plt):
    spatial_stages = [
        (name, output) for _, name, output in activations if output.ndim == 3
    ]
    channels_per_row = 8
    feature_rows = sum(
        (output.shape[-1] + channels_per_row - 1) // channels_per_row
        for _, output in spatial_stages
    )
    feature_fig, feature_axes = plt.subplots(
        feature_rows,
        channels_per_row,
        figsize=(2.4 * channels_per_row, 3.0 * feature_rows),
        squeeze=False,
    )
    feature_fig.subplots_adjust(left=0.08, top=0.95, hspace=1.0)
    feature_row_index = 0
    for stage_name, output in spatial_stages:
        stage_start_row = feature_row_index
        stage_rows = (output.shape[-1] + channels_per_row - 1) // channels_per_row
        for stage_row_index in range(stage_rows):
            axis_row = feature_axes[feature_row_index]
            first_channel = stage_row_index * channels_per_row
            for column_index, feature_axis in enumerate(axis_row):
                channel_index = first_channel + column_index
                if channel_index < output.shape[-1]:
                    feature_axis.imshow(output[..., channel_index], cmap="magma_r")
                    feature_axis.set_xticks([])
                    feature_axis.set_yticks([])
                else:
                    feature_axis.axis("off")
            feature_row_index += 1
        first_axis = feature_axes[stage_start_row, 0]
        last_axis = feature_axes[feature_row_index - 1, -1]
        first_bounds = first_axis.get_position()
        last_bounds = last_axis.get_position()
        left = first_bounds.x0 - 0.01
        right = last_bounds.x1 + 0.01
        bottom = last_bounds.y0 - 0.01
        top = first_bounds.y1 + 0.01
        feature_fig.add_artist(
            Rectangle(
                (left, bottom),
                right - left,
                top - bottom,
                transform=feature_fig.transFigure,
                fill=False,
                linewidth=2.5,
                edgecolor="black",
                clip_on=False,
            )
        )
        feature_fig.text(
            (left + right) / 2,
            top,
            stage_name,
            transform=feature_fig.transFigure,
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
    feature_fig.suptitle(
        f"Feature maps for {feature_source}",
        fontsize=24,
    )
    feature_fig
    return


@app.cell(hide_code=True)
def _(Dense, activations, feature_source, math, plt, pooled_output):
    dense_output = next(
        output for layer, _, output in activations if isinstance(layer, Dense)
    )
    probabilities = activations[-1][2]
    predicted_label = int(probabilities.argmax())
    pooled_side = math.isqrt(pooled_output.shape[0])

    output_fig, output_axes = plt.subplots(1, 3, figsize=(13, 3.5))
    output_axes[0].imshow(
        pooled_output[: pooled_side * pooled_side].reshape(pooled_side, pooled_side),
        cmap="viridis",
    )
    output_axes[0].set_title("Global average pooling output")
    output_axes[0].axis("off")
    output_axes[1].bar(range(10), dense_output)
    output_axes[1].set_title("Dense output (logits)")
    output_axes[1].set_xlabel("Class")
    output_axes[1].set_xticks(range(10))
    colors = [
        "tab:orange" if class_index == predicted_label else "tab:blue"
        for class_index in range(10)
    ]
    output_axes[2].bar(range(10), probabilities, color=colors)
    output_axes[2].set_title("Predicted probability by class")
    output_axes[2].set_xlabel("Class")
    output_axes[2].set_ylabel("Probability")
    output_axes[2].set_xticks(range(10))
    output_axes[2].set_ylim(0, 1)
    for probability_index, probability in enumerate(probabilities):
        output_axes[2].text(
            probability_index,
            float(probability) + 0.02,
            f"{float(probability):.2f}",
            ha="center",
            fontsize=7,
        )
    output_fig.suptitle(
        f"{feature_source.capitalize()} - predicted: {predicted_label} "
        "(predicted class's bar in orange)"
    )
    output_fig
    return


if __name__ == "__main__":
    app.run()
