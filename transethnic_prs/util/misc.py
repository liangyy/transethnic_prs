import warnings

import numpy as np
import jax.numpy as jnp

def check_if_using_float64():
    x = jnp.array(1.)
    if x.dtype is jnp.dtype('float64'):
        return
    else:
        warnings.warn(
            '''
            Using float32. May not have enough precision for convergence check.
            Do the following the set the precision: 
            > from jax.config import config
            > config.update("jax_enable_x64", True)
            '''
        )

def check_np_darray(npdarray, dim=1, check_squared=False):
    if dim > 2:
        raise ValueError('Only check dim = 1 or 2.')
    if not isinstance(npdarray, np.ndarray):        
        raise TypeError('Not np.darray.')
    if dim != len(npdarray.shape):
        raise ValueError('Dim does not match.')
    if dim == 1:
        return npdarray.shape[0]
    elif dim == 2:
        m, n = npdarray.shape
        if check_squared is True:
            if m != n:
                raise ValueError('Matrix is not squared.')
            return m
        else:
            return m, n
        
def merge_two_lists(l1, l2):
    out = []
    for i, j in zip(l1, l2):
        out.append(f'{i}_{j}')
    return out   

def scale_array_list(array_list, factor):
    return [ x * factor for x in array_list ]

def intersect_two_lists(l1, l2):
    s1 = set(l1)
    return list(s1.intersection(l2))
        