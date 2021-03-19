import jax.numpy as jnp
from jax import jit

@jit
def soft_thres_jax(x, thres):
    tmp = jnp.abs(x) - thres
    return jnp.where(tmp < 0, 0, jnp.sign(x) * tmp)

@jit
def calc_XXt_diag_jax(X):
    return jnp.power(X, 2).sum(axis=1)
    
@jit
def calc_Xy_jax(X, y):
    return X @ y

@jit
def mean_center_col_2d_jax(x):
    return x - x.mean(axis=0)

@jit
def mean_center_col_1d_jax(x):
    return x - x.mean()
