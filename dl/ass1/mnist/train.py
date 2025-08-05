import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from model import MLP


def init_opt(model: MLP, learning_rate) -> nnx.Optimizer:
    learning_rate = learning_rate
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=learning_rate))
    return optimizer


def loss_fn(model: MLP, batch):
    logits = model(batch[0])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch[1]
    ).mean()
    return loss


@nnx.jit
def train(model: MLP, optimizer: nnx.Optimizer, batch: jnp.ndarray) -> int:
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    optimizer.update(grads)

    return loss


@nnx.jit
def test(model: MLP, batch: jnp.ndarray):
    logits = jax.vmap(model)(batch[0])
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(preds == batch[1])

    return acc
