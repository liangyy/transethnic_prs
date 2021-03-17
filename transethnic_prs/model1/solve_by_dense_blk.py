import jax.numpy as jnp
import jax.lax
from jax import jit
import jax.ops

from transethnic_prs.util.misc import check_if_using_float64
from transethnic_prs.util.math import soft_thres_jax

'''
CAUTIOUS: we need float64 to have it more stable and converge 
using float32, sometimes, obj will increase
Do the following the set the precision: 
> from jax.config import config
> config.update("jax_enable_x64", True)
'''

def solve_by_dense_blk(Alist, blist, X, y, w1, w2, tol=1e-5, maxiter=1000):
    check_if_using_float64()
    n, p = X.shape
    # init
    beta = [ jnp.zeros(A_blk.shape[0]) for A_blk in Alist ]
    t = [ jnp.zeros(A_blk.shape[0]) for A_blk in Alist ]
    r = y.copy()
    XtX_diag = (X ** 2).sum(axis=0)
    # solve obj directly since beta = 0
    obj0 = (r.T @ r).sum()
    diff = tol + 1
    niter = 0
    while diff > tol and niter < maxiter:
        offset = 0
        beta_Abeta = 0
        beta_b = 0
        l1_beta = 0
        l2_beta = 0
        for blk, A_blk in enumerate(Alist):
            p_blk = A_blk.shape[0]
            _, (t[blk], beta[blk], r), _, _, (beta_Abeta, beta_b, l1_beta, l2_beta) = _dense_blk_update(
                val=[
                    (A_blk, blist[blk], X, XtX_diag), 
                    (t[blk], beta[blk], r), 
                    (w1, w2), 
                    offset, 
                    (beta_Abeta, beta_b, l1_beta, l2_beta)
                ]
            )
            offset += p_blk
        obj = _eval_obj_jax(beta_Abeta, beta_b, l1_beta, l2_beta, r, w1, w2)
        diff = obj0 - obj
        obj0 = obj
        niter += 1
    return beta, niter, diff

@jit
def _dense_blk_update(val):
    '''
    val = [
        (A_blk, blist[blk], X, XtX_diag), 
        (t[blk], beta[blk], r), 
        (w1, w2), 
        offset, 
        (beta_Abeta, beta_b, l1_beta, l2_beta)
    ]
    '''
    A = val[0][0]
    p = A.shape[0]
    beta_Abeta, beta_b, l1_beta, l2_beta = val[-1]
    val = jax.lax.fori_loop(
        0, p, _dense_update_jth, 
        init_val=val[:-1]
    )
    b, t, beta = val[0][1], val[1][0], val[1][1]
    beta_Abeta += beta.T @ t
    beta_b += b.T @ beta
    l1_beta += jnp.abs(beta).sum()
    l2_beta += jnp.power(beta, 2).sum()
    return val + [ (beta_Abeta, beta_b, l1_beta, l2_beta) ]
    
@jit
def _eval_obj_jax(beta_Abeta, beta_b, l1_beta, l2_beta, resid, w1, w2):
    return beta_Abeta - 2 * beta_b + resid.T @ resid + w1 * l1_beta + w2 * l2_beta

def _dense_update_jth(j, val):
    '''
    val = [ 
        (A_blk, blist[blk], X, XtX_diag), 
        (t[blk], beta[blk], r), 
        (w1, w2), 
        offset
    ]
    '''
    A, b, X, XtX_diag = val[0]
    t, beta, r = val[1]
    w1, w2 = val[2]
    offset = val[3]
    
    j_abs = j + offset
    beta_j = beta[j]
    Xj = X[:, j_abs]
    r_no_j = r + Xj * beta_j
    Ajj = A[j, j]
    Abeta_no_j = t[j] - Ajj * beta_j
    a = Ajj + XtX_diag[j_abs] + w2
    c = Abeta_no_j - b[j] - r_no_j.T @ Xj
    beta_j_new = - soft_thres_jax(2 * c, w1) / a / 2
    r = r_no_j - Xj * beta_j_new
    t = t - A[j, :] * (beta_j - beta_j_new)
    beta = jax.ops.index_update(beta, j, beta_j_new)
    
    val[1] = (t, beta, r)
    
    return val


