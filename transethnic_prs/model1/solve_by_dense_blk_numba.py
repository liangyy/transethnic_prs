import numba as nb
import numpy as np

from transethnic_prs.util.math_numba import soft_thres_numba

'''
A = X1.T @ X1
b = X1.T @ y1
Xt = X2.T
XtX_diag = diag(X2.T @ X2)
y = y2

Objective:

argmin_x x' A x - 2 b' x + offset * x' diag(A) x + || y2 - X2 x ||_2^2 + w1 ||x||_1 + w2 ||x||_2^2  
'''

def solve_by_dense_blk_numba(Alist, blist, Xtlist, y, XtX_diag_list, w1, w2, offset=0, tol=1e-5, maxiter=1000,
    init_beta=None, init_t=None, init_r=None):
    
    # init
    if init_beta is None:
        beta, t, r = _init_beta_as_zeros_list(Alist, y)
    else:
        beta, t, r = init_beta, init_t, init_r
    
    offset_list = [ offset * ai.diagonal() for ai in Alist ]
    
    diff = tol + 1
    niter = 0
    while diff > tol and niter < maxiter:
        diff = 0.
        for blk, A_blk in enumerate(Alist):
            p_blk = A_blk.shape[0]
            beta[blk], _, diff_, t[blk], r, _ = solve_by_dense_blk_numba_(
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
                XtX_diag=XtX_diag_list[blk]
            )
            if diff < diff_:
                diff = diff_
        niter += 1
    if diff <= tol:
        conv = 1
    else:
        conv = 0
    return beta, niter, diff, t, r, conv

# solve with X1.T @ X1
def solve_by_dense_one_blk_numba(A, b, Xt, y, XtX_diag, w1, w2, offset=0, tol=1e-5, maxiter=1000,
    init_beta=None, init_t=None, init_r=None):
    
    # init
    if init_beta is None:
        beta, t, r = _init_beta_as_zeros(A, y)
    else:
        beta, t, r = init_beta, init_t, init_r
    
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
        'XtX_diag': XtX_diag
    }
    return solve_by_dense_blk_numba_(**args)

# solve with X1
def solve_by_dense_one_blk_X_numba(X_info_list, b, Xt, y, XtX_diag, w1, w2, offset=0, tol=1e-5, maxiter=1000,
    init_beta=None, init_t=None, init_r=None):
    
    # init
    if init_beta is None:
        beta, t, r = _init_beta_as_zeros_X(X_info_list[0], y)
    else:
        beta, t, r, = init_beta, init_t, init_r
    
    offset = offset * X_info_list[1]
    
        
    args_ = { 
        'X1t': X_info_list[0],
        'XtX1_diag': X_info_list[1], 
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
        'XtX_diag': XtX_diag
    }
    # breakpoint()
    return solve_by_dense_blk_X_numba_(**args_)


# @nb.jit(nb.float64(nb.float64, nb.float64, nb.float64[::1]))
# def _eval_obj_lik_numba(beta_Abeta, beta_b, resid):
#     return beta_Abeta - 2 * beta_b + resid.T @ resid 
#
# @nb.jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64))
# def _eval_penalty(l1_beta, l2_beta, w1, w2):
#     return w1 * l1_beta + w2 * l2_beta

@nb.jit(nb.types.Tuple((nb.float64[::1], nb.int64, nb.float64, nb.float64[::1], nb.float64[::1], nb.int64))\
(nb.float64[:, ::1], \
nb.float64[::1], \
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
nb.float64[::1]))
def solve_by_dense_blk_X_numba_(X1t, XtX1_diag, b, Xt, y, w1, w2, offset, tol, maxiter, beta, t, r, XtX_diag):
    diff = tol + 1
    niter = 0
    while diff > tol and niter < maxiter:
        diff = 0.
        for j in range(X1t.shape[0]):
            beta_j = beta[j]
            Xj = Xt[j, :]
            X1j = X1t[j, :]
        
            if beta_j != 0:
                r = r + Xj * beta_j
                t = t - X1j * beta_j
        
            a = XtX1_diag[j] + XtX_diag[j] + w2 + offset[j]
            c = t.T @ X1j - b[j] - r.T @ Xj
        
            beta_j_new = - soft_thres_numba(2 * c, w1) / a / 2
        
            if beta_j_new != 0:
                r = r - Xj * beta_j_new
                t = t + X1j * beta_j_new
            
            diff_ = np.abs(beta[j] - beta_j_new)
            if diff < diff_:
                diff = diff_
                 
            beta[j] = beta_j_new
        
        niter += 1
    if diff <= tol:
        conv = 1
    else:
        conv = 0
    return beta, niter, diff, t, r, conv

@nb.jit(nb.types.Tuple((nb.float64[::1], nb.int64, nb.float64, nb.float64[::1], nb.float64[::1], nb.int64))\
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
nb.float64[::1]))
def solve_by_dense_blk_numba_(A, b, Xt, y, w1, w2, offset, tol, maxiter, beta, t, r, XtX_diag):
    diff = tol + 1
    niter = 0
    while diff > tol and niter < maxiter:
        diff = 0.
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
            
            diff_ = np.abs(beta[j] - beta_j_new)
            if diff < diff_:
                diff = diff_
            
            beta[j] = beta_j_new
        
        
        niter += 1
    if diff <= tol:
        conv = 1
    else:
        conv = 0
    return beta, niter, diff, t, r, conv

def _init_beta_as_zeros(A, y):
    beta = np.zeros(A.shape[0]) * 0.
    t = np.zeros(A.shape[0]) * 0.
    r = y - 0.
    return beta, t, r

def _init_beta_as_zeros_X(Xt, y):
    beta = np.zeros(Xt.shape[0]) * 0.
    t = np.zeros(Xt.shape[1]) * 0.
    r = y - 0.
    return beta, t, r

def _init_beta_as_zeros_list(Alist, y):
    beta = [ np.zeros(A_blk.shape[0]) * 0. for A_blk in Alist ]
    t = [ np.zeros(A_blk.shape[0]) * 0. for A_blk in Alist ]
    r = y - 0.
    return beta, t, r

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


