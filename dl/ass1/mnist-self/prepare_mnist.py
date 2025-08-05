import polars as pl
import io
from PIL import Image
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

total = 60_000

def get_df_train() -> pl.DataFrame:
    df = pl.read_parquet("./mnist/mnist/train-00000-of-00001.parquet")
    df = df.select(pl.col("image").struct.field("bytes"), pl.col("label").cast(pl.UInt8))

    return df

def get_df_test() -> pl.DataFrame:
    df = pl.read_parquet("./mnist/mnist/test-00000-of-00001.parquet")
    df = df.select(pl.col("image").struct.field("bytes"), pl.col("label").cast(pl.UInt8))

    return df


def prepare_data(df: pl.DataFrame):
    images = []
    labels = []
    for row in df.iter_rows():
        (binary, label) = row
        image_arr = Image.open(io.BytesIO(binary)).convert('L')

        images.append(image_arr)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    images = images.reshape(images.shape[0], -1)
    images = images.astype(jnp.float32) / 255.0

    return (np.array(images), np.array(labels))

def prepare_train_test():
    
    img_bytes, labels = prepare_data(get_df_train())
    np.savez("mnist_train.npz", images=img_bytes, labels=labels)

    img_bytes, labels = prepare_data(get_df_test())
    np.savez("mnist_test.npz", images=img_bytes, labels=labels)

def get_data(type) -> (jnp.ndarray, jnp.ndarray):
    if type == "train":
        data = np.load("mnist_train.npz")
    elif type == "test":
        data = np.load("mnist_test.npz")
    else:
        raise Exception("Valid values are only: ['train', 'test']")

    images = jnp.array(data["images"])
    labels = jnp.array(data["labels"])

    return images, labels

def get_batch(data, batch_size: int, index: int) -> (jnp.ndarray, jnp.ndarray):
    start = index * batch_size
    end = min(start + batch_size, data[0].shape[0])

    images = lax.slice(data[0], (start, 0), (end, data[0].shape[1]))
    labels = lax.slice(data[1], (start,), (end,))
    return images, labels
