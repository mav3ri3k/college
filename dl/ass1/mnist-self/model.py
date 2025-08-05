import jax
import jax.numpy as jnp
from functools import partial

class MLP():
    @staticmethod
    def params(rngs: jnp.ndarray):
        initializer = jax.nn.initializers.glorot_normal()

        return [
            [
                initializer(rngs, (28 * 28, 256)),
                jnp.zeros(256),
            ],
            [
                initializer(rngs, (256, 10)),
                jnp.zeros(10),
            ],
        ]

    @staticmethod
    @partial(jax.vmap, in_axes=[None, 0])
    def forward(params, x: jnp.ndarray) -> jnp.ndarray:
        """give value returned by fn"""

        x = jnp.dot(x, params[0][0]) + params[0][1]
        x = jax.nn.relu(x)
        x = jnp.dot(x, params[1][0]) + params[1][1]

        return x

    @staticmethod
    @jax.vmap
    def softmax_cross_entropy_with_integer_labels(logits: jnp.ndarray, label: int) -> jnp.ndarray:
        return -jax.nn.log_softmax(logits)[label]


    @staticmethod
    @jax.jit
    def update(params, grads, learning_rate):
        return [
            [
                p - learning_rate * g for p, g in zip(layer_p, layer_g)
            ]
            for layer_p, layer_g in zip(params, grads)
        ]
      
