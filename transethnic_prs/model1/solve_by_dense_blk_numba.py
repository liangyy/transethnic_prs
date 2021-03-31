import numba as nb
import numpy as np

from transethnic_prs.util.math_numba import soft_thres_numba

def solve_by_dense_blk_numba(Alist, blist, Xtlist, y, XtX_diag_list, w1, w2, offset=0, tol=1e-5, maxiter=1000,
    init_beta=None, init_t=None, init_r=None, init_obj_lik=None, 
    init_l1_beta=None, init_l2_beta=None):
    
    # init
    if init_beta is None:
        beta, t, r, obj_lik, l1_beta, l2_beta = _init_beta_as_zeros_list(Alist, y)
    else:
        beta, t, r, obj_lik, l1_beta, l2_beta = init_beta, init_t, init_r, init_obj_lik, init_l1_beta, init_l2_beta
    
    offset_list = [ offset * ai.diagonal() for ai in Alist ]
    obj0 = obj_lik + _eval_penalty(l1_beta, l2_beta, w1, w2) 
    diff = tol + 1
    niter = 0
    while diff > tol and niter < maxiter:
        beta_Abeta = 0.
        beta_b = 0.
        l1_beta = 0.
        l2_beta = 0.
        for blk, A_blk in enumerate(Alist):
            p_blk = A_blk.shape[0]
            beta[blk], _, _, t[blk], r, _, l1_beta_i, l2_beta_i, beta_Abeta_i, beta_b_i = solve_by_dense_blk_numba_(
                A=A_blk, 
                b=blist[blk], 
                Xt=Xtlist[blk], 
                y=y, 
                w1=w1, 
                w2=w2, 
                offset=offset_list[blk], 
                tol=1.0, 
                maxiter=1, 
                beta=beta[blk], 
                t=t[blk], 
                r=r, 
                obj_lik=0., 
                l1_beta=0., 
                l2_beta=0., 
                XtX_diag=XtX_diag_list[blk]
            )
            beta_Abeta += beta_Abeta_i
            beta_b += beta_b_i
            l1_beta += l1_beta_i
            l2_beta += l2_beta_i
        obj_lik = _eval_obj_lik_numba(beta_Abeta, beta_b, r)
        obj_penalty = _eval_penalty(l1_beta, l2_beta, w1, w2)
        obj = obj_lik + obj_penalty
        diff = obj0 - obj
        obj0 = obj
        niter += 1
    return beta, niter, diff, (t, r, obj_lik, l1_beta, l2_beta)

def solve_by_dense_one_blk_numba(A, b, Xt, y, XtX_diag, w1, w2, offset=0, tol=1e-5, maxiter=1000,
    init_beta=None, init_t=None, init_r=None, init_obj_lik=None, 
    init_l1_beta=None, init_l2_beta=None):
    
    # init
    if init_beta is None:
        beta, t, r, obj_lik, l1_beta, l2_beta = _init_beta_as_zeros(A, y)
    else:
        beta, t, r, obj_lik, l1_beta, l2_beta = init_beta, init_t, init_r, init_obj_lik, init_l1_beta, init_l2_beta
    
    offset = offset * A.diagonal()
    
    args = { 
        'A': A, 
        'b': b, 
        'Xt': Xt, 
        'y': y, 
        'w1': w1, 
        'w2': w2, 
        'offset': offset, 
        'tol': tol, 
        'maxiter': maxiter,
        'beta': beta, 
        't': t, 
        'r': r, 
        'obj_lik': obj_lik, 
        'l1_beta': l1_beta, 
        'l2_beta': l2_beta, 
        'XtX_diag': XtX_diag
    }
    return solve_by_dense_blk_numba_(**args)


@nb.jit(nb.float64(nb.float64, nb.float64, nb.float64[::1]))
def _eval_obj_lik_numba(beta_Abeta, beta_b, resid):
    return beta_Abeta - 2 * beta_b + resid.T @ resid 

@nb.jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64))
def _eval_penalty(l1_beta, l2_beta, w1, w2):
    return w1 * l1_beta + w2 * l2_beta

@nb.jit(nb.types.Tuple((nb.float64[::1], nb.int64, nb.float64, nb.float64[::1], nb.float64[::1], nb.float64, nb.float64, nb.float64, \
nb.float64, \
nb.float64))\
(nb.float64[:, ::1], \
nb.float64[::1], \
nb.float64[:, ::1], \
nb.float64[::1], \
nb.float64, \
nb.float64, \
nb.float64[::1], \
nb.float64, \
nb.int64, \
nb.float64[::1], \
nb.float64[::1], \
nb.float64[::1], \
nb.float64, \
nb.float64, \
nb.float64, \
nb.float64[::1]))
def solve_by_dense_blk_numba_(A, b, Xt, y, w1, w2, offset, tol, maxiter, beta, t, r, obj_lik, l1_beta, l2_beta, XtX_diag):
    obj0 = obj_lik + _eval_penalty(l1_beta, l2_beta, w1, w2) 
    diff = tol + 1
    niter = 0
    while diff > tol and niter < maxiter:
        beta_Abeta = 0.
        beta_b = 0.
        l1_beta = 0.
        l2_beta = 0.
        for j in range(A.shape[0]):
            beta_j = beta[j]
            # Xj = X[:, j]
            # transpose X for speed concern
            Xj = Xt[j, :]
        
            # r_no_j = r + Xj * beta_j
            if beta_j != 0:
                r = r + Xj * beta_j
        
            Ajj = A[j, j]
        
            # Abeta_no_j = t[j] - Ajj * beta_j
            if beta_j != 0:
                t = t - A[j, :] * beta_j
        
            a = Ajj + XtX_diag[j] + w2 + offset[j]
        
            # c = Abeta_no_j - b[j] - r_no_j.T @ Xj
            # c = Abeta_no_j - b[j] - r.T @ Xj
            c = t[j] - b[j] - r.T @ Xj
        
            beta_j_new = - soft_thres_numba(2 * c, w1) / a / 2
        
            # r = r_no_j - Xj * beta_j_new
            if beta_j_new != 0:
                r = r - Xj * beta_j_new
                t = t + A[j, :] * beta_j_new
            # t = t - A[j, :] * (beta_j - beta_j_new)
        
            beta[j] = beta_j_new
        
        beta_Abeta += beta.T @ t + (np.power(beta, 2) * offset).sum()  
        beta_b += b.T @ beta
        l1_beta += np.abs(beta).sum()
        l2_beta += np.power(beta, 2).sum()   
        obj_lik = _eval_obj_lik_numba(beta_Abeta, beta_b, r)
        obj_penalty = _eval_penalty(l1_beta, l2_beta, w1, w2)
        obj = obj_lik + obj_penalty
        diff = obj0 - obj
        obj0 = obj
        niter += 1
    return beta, niter, diff, t, r, obj_lik, l1_beta, l2_beta, beta_Abeta, beta_b

def _init_beta_as_zeros(A, y):
    beta = np.zeros(A.shape[0]) * 0.
    t = np.zeros(A.shape[0]) * 0.
    r = y - 0.
    # solve obj_lik directly since beta = 0
    obj_lik = (r.T @ r).sum()
    l1_beta = 0.
    l2_beta = 0.
    return beta, t, r, obj_lik, l1_beta, l2_beta

def _init_beta_as_zeros_list(Alist, y):
    beta = [ np.zeros(A_blk.shape[0]) * 0. for A_blk in Alist ]
    t = [ np.zeros(A_blk.shape[0]) * 0. for A_blk in Alist ]
    r = y - 0.
    # solve obj_lik directly since beta = 0
    obj_lik = (r.T @ r).sum()
    l1_beta = 0.
    l2_beta = 0.
    return beta, t, r, obj_lik, l1_beta, l2_beta

# @jit
# def _dense_blk_update(val):
#     '''
#     val = [
#         (A_blk, blist[blk], X[blk], XtX_diag[blk]), 
#         (t[blk], beta[blk], r), 
#         (w1, w2, offset),
#         (beta_Abeta, beta_b, l1_beta, l2_beta)
#     ]
#     '''
#     A = val[0][0]
#     p = A.shape[0]
#     beta_Abeta, beta_b, l1_beta, l2_beta = val[-1]
#     offset = val[2][2]
#     val = jax.lax.fori_loop(
#         0, p, _dense_update_jth, 
#         init_val=val[:-1]
#     )
#     b, t, beta = val[0][1], val[1][0], val[1][1]
#     beta_Abeta += beta.T @ t + (jnp.power(beta, 2) * offset).sum()
#     beta_b += b.T @ beta
#     l1_beta += jnp.abs(beta).sum()
#     l2_beta += jnp.power(beta, 2).sum() 
#     return val + [ (beta_Abeta, beta_b, l1_beta, l2_beta) ]
    
# @jit
# def _eval_obj_jax(beta_Abeta, beta_b, l1_beta, l2_beta, resid, w1, w2):
#     return beta_Abeta - 2 * beta_b + resid.T @ resid + w1 * l1_beta + w2 * l2_beta



# def _dense_update_jth(j, val):
#     '''
#     val = [ 
#         (A_blk, blist[blk], X[blk], XtX_diag[blk]), 
#         (t[blk], beta[blk], r), 
#         (w1, w2, offset)
#     ]
#     '''
#     A, b, X, XtX_diag = val[0]
#     t, beta, r = val[1]
#     w1, w2, offset = val[2]
# 
#     beta_j = beta[j]
#     # Xj = X[:, j]
#     # transpose X for speed concern
#     Xj = X[j, :]
# 
#     # r_no_j = r + Xj * beta_j
#     r = jnp.where(beta_j == 0, r, r + Xj * beta_j)
# 
#     Ajj = A[j, j]
# 
#     # Abeta_no_j = t[j] - Ajj * beta_j
#     t = jnp.where(beta_j == 0, t, t - A[j, :] * beta_j)
# 
#     a = Ajj + XtX_diag[j] + w2 + offset[j]
# 
#     # c = Abeta_no_j - b[j] - r_no_j.T @ Xj
#     # c = Abeta_no_j - b[j] - r.T @ Xj
#     c = t[j] - b[j] - r.T @ Xj
# 
#     beta_j_new = - soft_thres_jax(2 * c, w1) / a / 2
# 
#     # r = r_no_j - Xj * beta_j_new
#     r = jnp.where(beta_j_new == 0, r, r - Xj * beta_j_new) 
# 
#     # t = t - A[j, :] * (beta_j - beta_j_new)
#     t = jnp.where(beta_j_new == 0, t, t + A[j, :] * beta_j_new)
# 
#     beta = jax.ops.index_update(beta, j, beta_j_new)
# 
#     val[1] = (t, beta, r)
# 
#     return val


