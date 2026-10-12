"""Download and load the MNIST dataset from the Hugging Face Hub.

Images are decoded to (N, 28, 28, 1) float32 arrays normalized to [0, 1];
labels are one-hot encoded with `NUM_CLASSES` columns.
"""

import io

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = "ylecun/mnist"
NUM_CLASSES = 10


def _load_split(split: str, limit: int | None = None) -> tuple[jax.Array, jax.Array]:
    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"mnist/{split}-00000-of-00001.parquet",
    )
    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)

    images = np.stack(
        [
            np.array(Image.open(io.BytesIO(record["bytes"])), dtype=np.float32) / 255.0
            for record in table.column("image").to_pylist()
        ]
    )[..., None]
    labels = jnp.array(table.column("label").to_numpy())

    return jnp.array(images), jax.nn.one_hot(labels, NUM_CLASSES)


def load(
    n_train: int | None = None, n_test: int | None = None
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    """Returns ((x_train, y_train), (x_test, y_test))."""
    x_train, y_train = _load_split("train", limit=n_train)
    x_test, y_test = _load_split("test", limit=n_test)

    return (x_train, y_train), (x_test, y_test)
