import time
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X_np = np.linspace(-10, 10, 1000).reshape(-1, 1).astype(np.float32)
y_np = (3*X_np**2 + 2*X_np + 1 +
        np.random.normal(0, 10, X_np.shape)).astype(np.float32)

X, y = jnp.array(X_np), jnp.array(y_np)

split = int(0.8*len(X))
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(16)(x); x = nn.relu(x)
        x = nn.Dense(8)(x);  x = nn.relu(x)
        return nn.Dense(1)(x)          # linear output

model = MLP()
rng   = jax.random.key(0)
init  = lambda: model.init(rng, jnp.ones((1,1)))["params"]

opts = {
    "SGD"           : optax.sgd(1e-3),
    "SGD+Momentum"  : optax.sgd(1e-3, momentum=0.9),
    "RMSprop"       : optax.rmsprop(1e-3),
    "Adam"          : optax.adam(1e-3),
}

EPOCHS     = 400
BATCH_SIZE = 64
STEPS      = len(X_train) // BATCH_SIZE

@jax.jit
def mse(params, x, y):
    return ((model.apply({"params": params}, x) - y) ** 2).mean()

@jax.jit
def train_step(state, xb, yb):
    loss, grads = jax.value_and_grad(mse)(state.params, xb, yb)
    return state.apply_gradients(grads=grads), loss

def run(name, opt):
    params = init()
    state  = TrainState.create(apply_fn=model.apply,
                               params=params,
                               tx=opt)

    losses = []
    start  = time.perf_counter()
    for epoch in range(EPOCHS):
        perm = jax.random.permutation(jax.random.key(epoch), len(X_train))
        Xb, yb = X_train[perm], y_train[perm]
        epoch_loss = 0.0
        for step in range(STEPS):
            b = slice(step*BATCH_SIZE, (step+1)*BATCH_SIZE)
            state, loss = train_step(state, Xb[b], yb[b])
            epoch_loss += loss
        losses.append(float(epoch_loss/STEPS))

    wall = time.perf_counter() - start
    final = mse(state.params, X_test, y_test)
    print(f"{name:<12} | final loss {final:.4f} | time {wall:.2f}s")
    return state.params, losses

hist, preds = {}, {}
for name, opt in opts.items():
    params, losses = run(name, opt)
    hist[name]  = losses
    preds[name] = model.apply({"params": params}, X_test)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
for name, ls in hist.items():
    plt.plot(ls, label=name)
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Loss vs Epoch")
plt.yscale("log"); plt.legend(); plt.grid()

plt.subplot(1,2,2)
for name, yhat in preds.items():
    plt.scatter(y_test, yhat, alpha=0.6, label=name)
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, 'k--')
plt.xlabel("True value"); plt.ylabel("Predicted"); plt.title("Predicted vs True")
plt.legend(); plt.grid()

plt.tight_layout()
plt.show()
