import numpy as np

import transethnic_prs.util.math_numba as mn 
from transethnic_prs.model1.Model1Helper import *
import transethnic_prs.model1.solve_by_dense_blk_numba as mod1_mn 

# for debugging
import time
# END

# used in multithreading call
def kkt_beta_zero_per_blk_(args):
    return kkt_beta_zero_per_blk(*args)
def calc_varx_(args):
    return calc_varx(*args)
def solve_path_by_snplist__(args):
    idx = args['blk_idx']
    t0 = time.time()
    if args['message'] > 0:
        print(f'Working on block idx = {idx}', flush=True)
    del args['blk_idx']
    tmp = solve_path_by_snplist(**args)
    t1 = time.time()
    if args['message'] > 0:
        print('time1 = ', t1 - t0, flush=True)
    return tmp  # solve_path_by_snplist(**args)
# END

def kkt_beta_zero_per_blk(loader2, b, n_factor, varx, snps, y, alpha, w_dict):
    if not isinstance(alpha, list):
        alpha = [ alpha ]
    lambda_max = { w: [ 0 for i in range(len(alpha)) ] for w in w_dict.keys() }
    xt = mn.mean_center_col_2d_numba(loader2.load(snps)).T.copy()
    xy = mn.calc_Xy_numba(xt, y)
    for w in lambda_max.keys():
        for idx, a_ in enumerate(alpha):
            # print('DEBUG', a_, 2 * np.absolute(b * varx * n_factor + w * xy).max())
            lambda_max[w][idx] = max(
                lambda_max[w][idx], 
                2 * np.absolute(b * varx * n_factor + w * xy).max() / a_
            )
    return lambda_max
def calc_varx(loader, snps):
    return mn.calc_varx_numba(loader.load(snps))


def solve_by_snplist(snplist, w1, w2, y, loader1=None, loader2=None, gwas_sample_size=None, gwas_bhat=None, A=None, b=None, X2t=None, mode=None, offset=0, w=1, tol=1e-5, maxiter=1000, message=0,
    return_raw=False, 
    # the following options only for internal use
    init_beta=None, init_t=None, init_r=None, 
    XtX2_diag=None):
    
    printit = False
    if A is None:
        t0 = time.time()
        # TODO: remove .T
        x1 = loader1.load(snplist)
        nx, px = x1.shape
        if mode is not None:
            pass
        else:
            if px > nx:
                mode = 'X'
            else:
                mode = 'XtX'
        if mode == 'X':
            # work with genotype X
            X1t = mn.mean_center_col_2d_numba(x1).T.copy() / np.sqrt(nx - 1) * np.sqrt(gwas_sample_size - 1)  # np.sqrt((nx - 1) / (gwas_sample_size - 1))
            XtX1_diag = mn.calc_XXt_diag_numba(X1t)
            A = (X1t, XtX1_diag)
            b = gwas_bhat * XtX1_diag
        elif mode == 'XtX':
            # work with X.T @ X (covx)
            A = mn.calc_covx_numba(x1.T.copy()) * (gwas_sample_size - 1)
            b = gwas_bhat * A.diagonal()
        printit = True
        
    if X2t is None:
        # TODO: remove .T
        X2t = mn.mean_center_col_2d_numba(loader2.load(snplist)).T.copy()
    if XtX2_diag is None:
        XtX2_diag = mn.calc_XXt_diag_numba(X2t)
    if message > 1 and printit is True:
        print('IO takes ', time.time() - t0, flush=True)
        print('mode = ', mode, flush=True)
    
    sqrt_w = np.sqrt(w)
    if mode == 'XtX':
        beta, niter, diff, t, r, conv = mod1_mn.solve_by_dense_one_blk_numba(
            A, b, X2t * sqrt_w, y * sqrt_w, 
            init_beta=init_beta, 
            init_t=init_t, 
            init_r=init_r,
            w1=w1, w2=w2, tol=tol, maxiter=maxiter, offset=offset, XtX_diag=XtX2_diag * w
        )
    elif mode == 'X':
        beta, niter, diff, t, r, conv = mod1_mn.solve_by_dense_one_blk_X_numba(
            A, b, X2t * sqrt_w, y * sqrt_w, 
            init_beta=init_beta, 
            init_t=init_t, 
            init_r=init_r,
            w1=w1, w2=w2, tol=tol, maxiter=maxiter, offset=offset, XtX_diag=XtX2_diag * w
        )
    if return_raw is False:
        # beta = merge_list([ beta ])
        return beta, niter, diff
    else:
        return beta, niter, diff, (t, r, conv), (A, b, X2t, XtX2_diag, mode)
        
        
def solve_path_by_snplist(snplist, lambda_seq_dict, data_args, alpha_list=[ 0.5 ], offset_x_w_list=[ [ 0, 1 ] ], tol=1e-5, maxiter=1000, mode=None, message=0):
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
    conv_out = []
    for oi in range(len(offset_x_w_list)):
        beta_out.append([])
        niter_out.append([])
        tol_out.append([])
        conv_out.append([])
        for wi in range(len(offset_x_w_list[0])):
            offset, w = offset_x_w_list[oi][wi]
            beta_out[-1].append([])
            niter_out[-1].append([])
            tol_out[-1].append([])
            conv_out[-1].append([])
            for lambda_seq, alpha in zip(lambda_seq_dict[w], alpha_list):
            
                nlambda = len(lambda_seq)
                
                pp = len(snplist)
                # initialize the beta mat (p x nlambda)
                beta_mat = np.zeros((pp, nlambda))
                # initialize niter and maxiter records
                niter_vec, tol_vec, conv_vec = np.zeros(nlambda), np.zeros(nlambda), - np.ones(nlambda)
                beta_mat[:, 0] = np.zeros(pp)
                
                # initialize beta, t, r
                beta, t, r = None, None, None
                
                # loop over lambda sequence skipping the first, lambda_max 
                for idx, lam in enumerate(lambda_seq):
                    
                    if idx == 0:
                        continue
                    w1, w2 = alpha_lambda_to_w1_w2(alpha, lam)
                    t0 = time.time()
                    beta, niter_vec[idx], tol_vec[idx], (t, r, conv_vec[idx]), (A, b, X2t, XtX2_diag, mode) = solve_by_snplist(
                        w1=w1, w2=w2, snplist=snplist,
                        A=A, b=b, X2t=X2t,
                        tol=tol, maxiter=maxiter, offset=offset, w=w,
                        init_beta=beta, init_t=t, init_r=r, 
                        XtX2_diag=XtX2_diag,
                        return_raw=True,
                        mode=mode,
                        message=message,
                        **data_args
                    )
                    tt = time.time() - t0
                    if message > 2:
                        print(f'w1 = {w1}, w2 = {w2}, time = {tt}', flush=True)
                    beta_mat[:, idx] = beta  
                beta_out[-1][-1].append(beta_mat)
                niter_out[-1][-1].append(niter_vec)
                tol_out[-1][-1].append(tol_vec)
                conv_out[-1][-1].append(conv_vec)
    
    return beta_out, niter_out, tol_out, conv_out
