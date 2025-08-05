# train.py
import jax
import jax.numpy as jnp
from functools import partial

# ---------- helpers -------------------------------------------------
def softmax(logits):
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = jnp.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)

def loss_and_grads(params, X, y_true):
    W1, b1 = params[0]
    W2, b2 = params[1]

    z1 = X @ W1 + b1               
    a1 = jnp.maximum(z1, 0.0)      
    logits = a1 @ W2 + b2          

    p = softmax(logits)
    loss = -jnp.log(p[jnp.arange(p.shape[0]), y_true]).mean()

    dlogits = p.at[jnp.arange(p.shape[0]), y_true].add(-1) / p.shape[0]  # (B,10)

    dW2 = a1.T @ dlogits                 
    db2 = dlogits.sum(axis=0)            

    da1 = dlogits @ W2.T                 
    dz1 = da1 * (z1 > 0)                 
    dW1 = X.T @ dz1                      
    db1 = dz1.sum(axis=0)                

    grads = [[dW1, db1], [dW2, db2]]
    return loss, grads

@partial(jax.jit, static_argnums=0)
def train(model, batch, learning_rate, params):
    X, y = batch
    loss, grads = loss_and_grads(params, X, y)
    params = model.update(params, grads, learning_rate)
    return loss, params

@partial(jax.jit, static_argnums=0)
def test(model, batch, params):
    X, y = batch
    logits = model.forward(params, X)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
