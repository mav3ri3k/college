import jax
import jax.numpy as jnp
from functools import partial

from model import MLP

def train(model: MLP, batch: jnp.ndarray, learning_rate, params) -> int:
    def loss_fn(params, x):
        pred = model.forward(params, batch['input'])
        loss = batch['output'] - pred

        return loss[0]

    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    print(loss, grads)
    params = model.update(params, grads, learning_rate)

    return loss, params
