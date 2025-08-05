from train import train
from model import MLP
import util
import jax
import jax.numpy as jnp

learning_rate = 1

key = jax.random.PRNGKey(0)
model = MLP()
params = model.params(key)

batch = {'input': jnp.array([0.801]), 'output': jnp.array([0.5])}

loss, params = train(model, batch, learning_rate, params)

print(loss, params)
