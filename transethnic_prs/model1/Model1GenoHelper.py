import numpy as np

import transethnic_prs.util.math_jax as mj 
from transethnic_prs.model1.Model1Helper import *
from transethnic_prs.model1.solve_by_dense_blk import solve_by_dense_blk

# used in multithreading call
def kkt_beta_zero_per_blk_(args):
    return kkt_beta_zero_per_blk(*args)
def calc_varx_(args):
    return calc_varx(*args)
def solve_path_by_snplist__(args):
    return solve_path_by_snplist(**args)
# END

def kkt_beta_zero_per_blk(loader2, b, n_factor, varx, snps, y, alpha):
    lambda_max = 0
    xt = mj.mean_center_col_2d_jax(loader2.load(snps)).T
    xy = mj.calc_Xy_jax(xt, y)
    lambda_max = max(
        lambda_max, 
        2 * np.absolute(b * varx * n_factor + xy).max() / alpha
    )
    return lambda_max
def calc_varx(loader, snps):
    return mj.calc_varx_jax(loader.load(snps))


def solve_by_snplist(snplist, w1, w2, y, loader1=None, loader2=None, gwas_n_factor=None, gwas_bhat=None, A=None, b=None, X2t=None, offset=0, tol=1e-5, maxiter=1000,
    return_raw=False, 
    # the following options only for internal use
    init_beta=None, init_t=None, init_r=None, 
    init_obj_lik=None, init_l1_beta=None, init_l2_beta=None,
    XtX2_diag=None):
    
    if A is None:
        x1t = mj.mean_center_col_2d_jax(loader1.load(snplist)).T
        A = mj.calc_covx_jax(x1t) * gwas_n_factor
    if b is None:    
        b = gwas_bhat * A.diagonal()
    if X2t is None:
        X2t = mj.mean_center_col_2d_jax(loader2.load(snplist)).T
        
    betalist, niter, diff, (tlist, rlist, obj_lik, l1_beta, l2_beta), XtX2_diag = solve_by_dense_blk(
        [ A ], [ b ], [ X2t ], y, 
        init_beta=init_beta, 
        init_t=init_t, 
        init_r=init_r,
        init_obj_lik=init_obj_lik,
        init_l1_beta=init_l1_beta, 
        init_l2_beta=init_l2_beta,
        XtX_diag_list=XtX2_diag,
        w1=w1, w2=w2, tol=tol, maxiter=maxiter, offset=offset
    )
    if return_raw is False:
        beta = merge_list(betalist)
        return beta, niter, diff
    else:
        return betalist, niter, diff, (tlist, rlist, obj_lik, l1_beta, l2_beta), (A, b, X2t, XtX2_diag)
        
        
def solve_path_by_snplist(snplist, lambda_seq, data_args, alpha=0.5, offset=0, tol=1e-5, maxiter=1000):
    
    nlambda = len(lambda_seq)
    
    pp = len(snplist)
    # initialize the beta mat (p x nlambda)
    beta_mat = np.zeros((pp, nlambda))
    # initialize niter and maxiter records
    niter_vec, tol_vec = np.zeros(nlambda), np.zeros(nlambda)
    beta_mat[:, 0] = np.zeros(pp)
    # initialize beta, t, r
    betalist, tlist, rlist, obj_lik, l1_beta, l2_beta = None, None, None, None, None, None
    A, b, X2t, XtX2_diag = None, None, None, None
    # loop over lambda sequence skipping the first, lambda_max 
    for idx, lam in enumerate(lambda_seq):
        # print('working on block = ', i, 'idx = ', idx)
        if idx == 0:
            continue
        w1, w2 = alpha_lambda_to_w1_w2(alpha, lam)
        # if idx >= 36:
        #     breakpoint()
        betalist, niter_vec[idx], tol_vec[idx], (tlist, rlist, obj_lik, l1_beta, l2_beta), (A, b, X2t, XtX2_diag) = solve_by_snplist(
            w1=w1, w2=w2, snplist=snplist,
            A=A, b=b, X2t=X2t,
            tol=tol, maxiter=maxiter, offset=offset,
            init_beta=betalist, init_t=tlist, init_r=rlist, 
            init_obj_lik=obj_lik, init_l1_beta=l1_beta, init_l2_beta=l2_beta,
            XtX2_diag=XtX2_diag,
            return_raw=True,
            **data_args
        )
        beta_mat[:, idx] = merge_list(betalist)
    
    return beta_mat, niter_vec, tol_vec