import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import matplotlib.pyplot as plt
import numpy as onp

X = jnp.array([[98.6, 0.2],
               [99.5, 0.9],
               [98.7, 0.8],
               [99.0, 0.3],
               [100.2, 0.7],
               [98.4, 0.6]], dtype=jnp.float32)

y = jnp.array([[0],
               [1],
               [1],
               [0],
               [1],
               [0]], dtype=jnp.float32)

mu, sigma = X.mean(0), X.std(0)
Xn = (X - mu) / sigma

class MLP(nn.Module):
    hidden: int = 4
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        return nn.Dense(1)(x)

model = MLP()
rng = jax.random.key(0)
params = model.init(rng, jnp.ones((1, 2)))["params"]

@jax.jit
def loss_fn(params, x, y):
    logits = model.apply({"params": params}, x)
    return optax.sigmoid_binary_cross_entropy(logits, y).mean()

@jax.jit
def accuracy(params, x, y):
    logits = model.apply({"params": params}, x)
    return ((logits > 0.5) == y).mean()

optimizer = optax.adam(1e-2)
state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=optimizer
)

@jax.jit
def train_step(state, x, y):
    grads = jax.grad(loss_fn)(state.params, x, y)
    return state.apply_gradients(grads=grads)

epochs = 500
loss_hist, acc_hist = [], []

for epoch in range(epochs):
    state = train_step(state, Xn, y)
    if epoch % 10 == 0 or epoch == epochs - 1:
        loss = loss_fn(state.params, Xn, y)
        acc  = accuracy(state.params, Xn, y)
        loss_hist.append(float(loss))
        acc_hist.append(float(acc))
        print(f"epoch {epoch:3d}  loss={loss:.4f}  acc={acc:.3f}")

plt.figure(figsize=(6,4))
h = 200
xx, yy = onp.meshgrid(onp.linspace(X[:,0].min()-1, X[:,0].max()+1, h),
                      onp.linspace(X[:,1].min()-1, X[:,1].max()+1, h))
grid = jnp.c_[xx.ravel(), yy.ravel()]
grid_n = (grid - mu) / sigma
zz = nn.sigmoid(model.apply({"params": state.params}, grid_n)).reshape(xx.shape)

plt.contourf(xx, yy, zz, levels=[0, 0.5, 1], cmap="coolwarm", alpha=.3)
plt.scatter(X[:,0], X[:,1], c=y.ravel(), cmap="coolwarm", edgecolor="k")
plt.xlabel("Temperature (°F)")
plt.ylabel("Inflammation")
plt.title("Learned decision boundary")
plt.colorbar(label="Flu prob")
plt.tight_layout()

@jax.jit
def hidden_rep(params, x):
    x = nn.Dense(4).apply({"params": params["Dense_0"]}, x)
    return nn.relu(x)

H = hidden_rep(state.params, Xn)          # (6, 4)

H = H - H.mean(0)                         
cov = jnp.cov(H.T)                        
w, V = jnp.linalg.eigh(cov)               
idx = jnp.argsort(w)[::-1][:2]            
proj = H @ V[:, idx]                      

plt.figure(figsize=(5,4))
plt.scatter(proj[:,0], proj[:,1], c=y.ravel(), cmap="coolwarm", edgecolor="k")
plt.title("Hidden-layer 2-D projection")
plt.xlabel("PC-1"); plt.ylabel("PC-2")
plt.tight_layout()
plt.show()
