import jax
import jax.numpy as jnp
import flax.nnx as nnx


class MLP(nnx.Module):
    def __init__(self, dropout_factor, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(28 * 28, 256, rngs=rngs)
        self.linear2 = nnx.Linear(256, 10, rngs=rngs)

    def __call__(self, x: jnp.ndarray):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)

        return x
