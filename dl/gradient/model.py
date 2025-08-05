import jax
import jax.numpy as jnp
from functools import partial

class MLP():
    @staticmethod
    def params(rngs: jnp.ndarray):
        initializer = jax.nn.initializers.constant(0.801)

        return [
            initializer(rngs, (1))
        ]

    @staticmethod
    def forward(params, x: jnp.ndarray) -> jnp.ndarray:
        """give value returned by fn"""
        x = jax.nn.sigmoid(x)
        return x

    @staticmethod
    @jax.jit
    def update(params, grads, learning_rate):
        print("update")
        print(params, grads)
        return [
            [
                p - learning_rate * g for p, g in zip(layer_p, layer_g)
            ]
            for layer_p, layer_g in zip(params, grads)
        ]
      
