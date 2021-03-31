import time

import numpy as np
from jax import local_device_count

import transethnic_prs.util.math_jax as mj 
from transethnic_prs.model1.Model1Helper import *
from transethnic_prs.model1.solve_by_dense_blk import solve_by_dense_blk

# used in multithreading call
def kkt_beta_zero_per_blk_(args):
    return kkt_beta_zero_per_blk(*args)
def calc_varx_(args):
    return calc_varx(*args)
def solve_path_by_snplist__(args):
    idx = args['blk_idx']
    t0 = time.time()
    print(f'Working on block idx = {idx}', flush=True)
    del args['blk_idx']
    tmp = solve_path_by_snplist(**args)
    ncpu = local_device_count()
    t1 = time.time()
    print('call again')
    _ = solve_path_by_snplist(**args)
    t2 = time.time()
    print('time1 = ', t1 - t0, ' time2 = ', t2 - t1, ' ncpu = ', ncpu, flush=True)
    return tmp
# END

def kkt_beta_zero_per_blk(loader2, b, n_factor, varx, snps, y, alpha):
    if not isinstance(alpha, list):
        alpha = [ alpha ]
    lambda_max = [ 0 for i in range(len(alpha)) ]
    xt = mj.mean_center_col_2d_jax(loader2.load(snps)).T
    xy = mj.calc_Xy_jax(xt, y)
    for idx, a_ in enumerate(alpha):
        lambda_max[idx] = max(
            lambda_max, 
            2 * np.absolute(b * varx * n_factor + xy).max() / a_
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
    
    t0 = time.time()
    printit = False
    if A is None:
        x1t = mj.mean_center_col_2d_jax(loader1.load(snplist)).T
        A = mj.calc_covx_jax(x1t) * gwas_n_factor
        printit = True
    if b is None:    
        b = gwas_bhat * A.diagonal()
    if X2t is None:
        X2t = mj.mean_center_col_2d_jax(loader2.load(snplist)).T
    t1 = time.time()
    if printit is True:
        print('IO takes ', t1 - t0, flush=True)
        
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
        
        
def solve_path_by_snplist(snplist, lambda_seq_list, data_args, alpha_list=[ 0.5 ], offset_list=[ 0 ], tol=1e-5, maxiter=1000):
    '''
    beta_out = 
    [ # for each alpha
      [ # for each offset 
        a beta matrix (nsnp x nlambda)
      ]
    ]
    '''
    
    # init data (need IO)
    A, b, X2t, XtX2_diag = None, None, None, None
    
    # collect results for each alpha
    beta_out = []
    niter_out = []
    tol_out = []
    for lambda_seq, alpha in zip(lambda_seq_list, alpha_list):
        
        # collect results for alpha = alpha across all offsets
        beta_out_alpha = []
        niter_out_alpha = []
        tol_out_alpha = []
        for offset in offset_list:
            
            nlambda = len(lambda_seq)
            
            pp = len(snplist)
            # initialize the beta mat (p x nlambda)
            beta_mat = np.zeros((pp, nlambda))
            # initialize niter and maxiter records
            niter_vec, tol_vec = np.zeros(nlambda), np.zeros(nlambda)
            beta_mat[:, 0] = np.zeros(pp)
            
            # initialize beta, t, r
            betalist, tlist, rlist, obj_lik, l1_beta, l2_beta = None, None, None, None, None, None
            
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
            beta_out_alpha.append(beta_mat)
            niter_out_alpha.append(niter_vec)
            tol_out_alpha.append(tol_vec)
        beta_out.append(beta_out_alpha)
        niter_out.append(niter_out_alpha)
        tol_out.append(tol_out_alpha)
    
    return beta_out, niter_out, tol_out
