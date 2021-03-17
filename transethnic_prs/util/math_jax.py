import jax.numpy as jnp
from jax import jit

@jit
def soft_thres_jax(x, thres):
    tmp = jnp.abs(x) - thres
    return jnp.where(tmp < 0, 0, jnp.sign(x) * tmp)
