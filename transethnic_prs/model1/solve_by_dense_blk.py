import jax.numpy as jnp
import jax.lax
from jax import jit
import jax.ops

from transethnic_prs.util.misc import check_if_using_float64
from transethnic_prs.util.math_jax import soft_thres_jax, calc_XXt_diag_jax

'''
CAUTIOUS: we need float64 to have it more stable and converge 
using float32, sometimes, obj will increase
Do the following the set the precision: 
> from jax.config import config
> config.update("jax_enable_x64", True)
'''

def solve_by_dense_blk(Alist, blist, Xtlist, y, w1, w2, offset=0, tol=1e-5, maxiter=1000,
    init_beta=None, init_t=None, init_r=None, init_obj_lik=None, 
    init_l1_beta=None, init_l2_beta=None, XtX_diag_list=None):
    
    check_if_using_float64()
    
    # init
    if init_beta is None:
        beta, t, r, obj_lik, l1_beta, l2_beta = _init_beta_as_zeros(Alist, y)
    else:
        beta, t, r, obj_lik, l1_beta, l2_beta = init_beta, init_t, init_r, init_obj_lik, init_l1_beta, init_l2_beta
    if XtX_diag_list is None:
        XtX_diag_list = [ calc_XXt_diag_jax(Xi) for Xi in Xtlist ]
    offset_list = [ offset * ai.diagonal() for ai in Alist ]
    obj0 = obj_lik + _eval_penalty(l1_beta, l2_beta, w1, w2) 
    diff = tol + 1
    niter = 0
    while diff > tol and niter < maxiter:
        beta_Abeta = 0
        beta_b = 0
        l1_beta = 0
        l2_beta = 0
        for blk, A_blk in enumerate(Alist):
            p_blk = A_blk.shape[0]
            _, (t[blk], beta[blk], r), _, (beta_Abeta, beta_b, l1_beta, l2_beta) = _dense_blk_update(
                val=[
                    (A_blk, blist[blk], Xtlist[blk], XtX_diag_list[blk]), 
                    (t[blk], beta[blk], r), 
                    (w1, w2, offset_list[blk]), 
                    (beta_Abeta, beta_b, l1_beta, l2_beta)
                ]
            )
        obj_lik = _eval_obj_lik_jax(beta_Abeta, beta_b, r)
        obj_penalty = _eval_penalty(l1_beta, l2_beta, w1, w2)
        obj = obj_lik + obj_penalty
        diff = obj0 - obj
        obj0 = obj
        niter += 1
    return beta, niter, diff, (t, r, obj_lik, l1_beta, l2_beta), XtX_diag_list

@jit
def _init_beta_as_zeros(Alist, y):
    beta = [ jnp.zeros(A_blk.shape[0]) for A_blk in Alist ]
    t = [ jnp.zeros(A_blk.shape[0]) for A_blk in Alist ]
    r = y - 0.
    # solve obj_lik directly since beta = 0
    obj_lik = (r.T @ r).sum()
    l1_beta = 0
    l2_beta = 0
    return beta, t, r, obj_lik, l1_beta, l2_beta

@jit
def _dense_blk_update(val):
    '''
    val = [
        (A_blk, blist[blk], X[blk], XtX_diag[blk]), 
        (t[blk], beta[blk], r), 
        (w1, w2, offset),
        (beta_Abeta, beta_b, l1_beta, l2_beta)
    ]
    '''
    A = val[0][0]
    p = A.shape[0]
    beta_Abeta, beta_b, l1_beta, l2_beta = val[-1]
    offset = val[2][2]
    val = jax.lax.fori_loop(
        0, p, _dense_update_jth, 
        init_val=val[:-1]
    )
    b, t, beta = val[0][1], val[1][0], val[1][1]
    beta_Abeta += beta.T @ t + (jnp.power(beta, 2) * offset).sum()
    beta_b += b.T @ beta
    l1_beta += jnp.abs(beta).sum()
    l2_beta += jnp.power(beta, 2).sum() 
    return val + [ (beta_Abeta, beta_b, l1_beta, l2_beta) ]
    
# @jit
# def _eval_obj_jax(beta_Abeta, beta_b, l1_beta, l2_beta, resid, w1, w2):
#     return beta_Abeta - 2 * beta_b + resid.T @ resid + w1 * l1_beta + w2 * l2_beta

@jit
def _eval_obj_lik_jax(beta_Abeta, beta_b, resid):
    return beta_Abeta - 2 * beta_b + resid.T @ resid 

@jit
def _eval_penalty(l1_beta, l2_beta, w1, w2):
    return w1 * l1_beta + w2 * l2_beta

def _dense_update_jth(j, val):
    '''
    val = [ 
        (A_blk, blist[blk], X[blk], XtX_diag[blk]), 
        (t[blk], beta[blk], r), 
        (w1, w2, offset)
    ]
    '''
    A, b, X, XtX_diag = val[0]
    t, beta, r = val[1]
    w1, w2, offset = val[2]
    
    beta_j = beta[j]
    # Xj = X[:, j]
    # transpose X for speed concern
    Xj = X[j, :]
    
    # r_no_j = r + Xj * beta_j
    r = jnp.where(beta_j == 0, r, r + Xj * beta_j)
    
    Ajj = A[j, j]
    
    # Abeta_no_j = t[j] - Ajj * beta_j
    t = jnp.where(beta_j == 0, t, t - A[j, :] * beta_j)
    
    a = Ajj + XtX_diag[j] + w2 + offset[j]
    
    # c = Abeta_no_j - b[j] - r_no_j.T @ Xj
    # c = Abeta_no_j - b[j] - r.T @ Xj
    c = t[j] - b[j] - r.T @ Xj
    
    beta_j_new = - soft_thres_jax(2 * c, w1) / a / 2
    
    # r = r_no_j - Xj * beta_j_new
    r = jnp.where(beta_j_new == 0, r, r - Xj * beta_j_new) 
    
    # t = t - A[j, :] * (beta_j - beta_j_new)
    t = jnp.where(beta_j_new == 0, t, t + A[j, :] * beta_j_new)
    
    beta = jax.ops.index_update(beta, j, beta_j_new)
    
    val[1] = (t, beta, r)
    
    return val


