import polars as pl
import io
from PIL import Image
import jax.numpy as jnp

total = 60_000

def get_df_train() -> pl.DataFrame:
    df = pl.read_parquet("./mnist/mnist/train-00000-of-00001.parquet")
    df = df.select(pl.col("image").struct.field("bytes"), pl.col("label").cast(pl.UInt8))

    return df
    
def get_batch(df: pl.DataFrame, batch_size: int, index: int) -> (jnp.ndarray, jnp.ndarray):
    images = []
    labels = []
    for j in range(batch_size):
        row = df.row(batch_size * index + j)
        (binary, label) = row
        image_arr = Image.open(io.BytesIO(binary)).convert('L')

        images.append(image_arr)
        labels.append(label)

    return (jnp.array(images), jnp.array(labels))

batch = get_batch(get_df_train(), 5, 5)

