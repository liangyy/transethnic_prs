'''

Model1 for trans-ethnic PRS.

Model/loss:
  || y1 - X1 * beta ||_2^2 +  || y - X * beta ||_2^2
  
Penalization: 
  w1 || beta ||_1 + w2 || beta ||_2^2

Where: 
  X1 and y1 are not available. 
  Instead, we know 
    X1'X1 = (N1 - 1) * Rhat1
    X1'y1 = (N1 - 1) * diag(Rhat1) * bhat1
    
So, the problem being solved is actually:
  argmin_beta 
    beta' * A * beta - 2 * b' * beta + 
    || y - X * beta ||_2^2 + 
    w1 || beta ||_1 + w2 || beta ||_2^2
, where
  A = (N1 - 1) Rhat1
  b = (N1 - 1) diag(Rhat1) bhat1

'''
import numpy as np
from transethnic_prs.model1.solve_by_dense_blk import solve_by_dense_blk
from transethnic_prs.util.misc import check_np_darray
from transethnic_prs.util.math_jax import calc_XXt_diag_jax, calc_Xy_jax, mean_center_col_2d_jax, mean_center_col_1d_jax

class Model1Blk:
    '''
    We take specific data structure for A and b.
    We assume A is block-wise diagonal.
    Alist = [ A1, ..., Ak ]
    blist = [ b1, ..., bk ]
    , where Ai is the ith squared matrix along the diagonal of A, 
    i.e. Ai = Xi'Xi
    and bi is the corresponding Xi'y
    CAUTION: we center Xlist and y
    '''
    def __init__(self, Alist, blist, Xlist, y):
        '''
        self.Alist
        self.blist
        self.Xtlist
        self.y
        self.p
        '''
        na_list = self._set_a_and_b(Alist, blist)
        px_list = self._set_x_and_y(Xlist, y)
        if not self._list_is_equal(na_list, px_list):
            raise ValueError('Different number of features between Alist and Xlist.')
        self.p = np.array(na_list).sum()
        self.XtX_diag_list = [ calc_XXt_diag_jax(Xi) for Xi in self.Xtlist ] 
    def _set_a_and_b(self, Alist, blist):
        if len(Alist) != len(blist):
            raise ValueError('Alist and blist have different length.')
        na_list = []
        for i in range(len(Alist)):
            na = check_np_darray(Alist[i], dim=2, check_squared=True)
            nb = check_np_darray(blist[i], dim=1)
            if na != nb:
                raise ValueError(f'The {i}th element in Alist and blist has un-matched shape {na} != {nb}.')
            na_list.append(na)
        self.Alist = Alist
        self.blist = blist
        return na_list
    def _set_x_and_y(self, xlist, y):
        if len(xlist) == 0:
            raise ValueError('xlist should have at least one element.')
        nx, px = check_np_darray(xlist[0], dim=2)
        px_list = [ px ]
        for i in range(1, len(xlist)):
            nx_, px_ = check_np_darray(xlist[i], dim=2)
            if nx != nx_:
                raise ValueError('Number of rows do not match in xlist.')
            px_list.append(px_)
        ny = check_np_darray(y, dim=1)
        if nx != ny:
            raise ValueError(f'X.shape[0] != y.shape[0].')
        # transpose X for speed concern 
        self.Xtlist = [ mean_center_col_2d_jax(x).T for x in xlist ]
        self.y = mean_center_col_1d_jax(y)   
        return px_list 
    @staticmethod
    def _list_is_equal(l1, l2):
        if len(l1) != len(l2):
            raise ValueError('l1 and l2 have different number of elements.')
        for n1, n2 in zip(l1, l2):
            if n1 != n2:
                return False
        return True
    @staticmethod
    def _get_lambda_seq(lambda_max, nlambda, ratio_lambda):
        lambda_min = lambda_max / ratio_lambda
        return np.exp(np.linspace(np.log(lambda_max), np.log(lambda_min), num=nlambda))
    @staticmethod
    def _alpha_lambda_to_w1_w2(alpha, lambda_):
        '''
        w1 = lambda * alpha
        w2 = lambda * (1 - alpha) / 2
        '''
        return lambda_ * alpha, lambda_ * (1 - alpha) / 2
    @staticmethod
    def _solve_path_param_sanity_check(alpha, nlambda, ratio_lambda):
        if alpha > 1 or alpha < 0:
            raise ValueError('Only alpha in [0, 1] is acceptable.')
        if not isinstance(nlambda, int) or nlambda < 1:
            raise ValueError('nlambda needs to be integer and nlambda >= 1')
        if not isinstance(nlambda, int) or ratio_lambda <= 1:
            raise ValueError('ratio_lambda needs to be integer and ratio_lambda > 1')
    def kkt_beta_zero(self, alpha):
        lambda_max = 0
        for b, xt in zip(self.blist, self.Xtlist):
            xy = calc_Xy_jax(xt, self.y)
            lambda_max = max(
                lambda_max, 
                2 * np.absolute(b + xy).max() / alpha
            )
        return lambda_max
    @staticmethod
    def _merge_list(ll):
        return np.concatenate(ll, axis=0)
    def solve(self, w1, w2, tol=1e-5, maxiter=1000, return_raw=False, 
        # the following options only for internal use
        init_beta=None, init_t=None, init_r=None, 
        init_obj_lik=None, init_l1_beta=None, init_l2_beta=None,
        XtX_diag_list=None
    ):
        betalist, niter, diff, (tlist, rlist, obj_lik, l1_beta, l2_beta) = solve_by_dense_blk(
            self.Alist, self.blist, self.Xtlist, self.y, 
            init_beta=init_beta, 
            init_t=init_t, 
            init_r=init_r,
            init_obj_lik=init_obj_lik,
            init_l1_beta=init_l1_beta, 
            init_l2_beta=init_l2_beta,
            XtX_diag_list=XtX_diag_list,
            w1=w1, w2=w2, tol=tol, maxiter=maxiter
        )
        if return_raw is False:
            beta = self._merge_list(betalist)
            return beta, niter, diff
        else:
            return betalist, niter, diff, (tlist, rlist, obj_lik, l1_beta, l2_beta)
    def solve_by_blk(self, idx, w1, w2, tol=1e-5, maxiter=1000, return_raw=False, 
        # the following options only for internal use
        init_beta=None, init_t=None, init_r=None, 
        init_obj_lik=None, init_l1_beta=None, init_l2_beta=None,
        XtX_diag_list=None
    ):
        betalist, niter, diff, (tlist, rlist, obj_lik, l1_beta, l2_beta) = solve_by_dense_blk(
            self.Alist[idx:(idx + 1)], self.blist[idx:(idx + 1)], self.Xtlist[idx:(idx + 1)], self.y, 
            init_beta=init_beta, 
            init_t=init_t, 
            init_r=init_r,
            init_obj_lik=init_obj_lik,
            init_l1_beta=init_l1_beta, 
            init_l2_beta=init_l2_beta,
            XtX_diag_list=XtX_diag_list,
            w1=w1, w2=w2, tol=tol, maxiter=maxiter
        )
        if return_raw is False:
            beta = self._merge_list(betalist)
            return beta, niter, diff
        else:
            return betalist, niter, diff, (tlist, rlist, obj_lik, l1_beta, l2_beta)
    def solve_path_by_blk(self, alpha=0.5, tol=1e-5, maxiter=1000, nlambda=100, ratio_lambda=100):
        '''
        Same info as solve_path.
        But here we solve each block one at a time and combine at the end.
        '''
        
        # check input parameters
        self._solve_path_param_sanity_check(alpha, nlambda, ratio_lambda)
        
        
        lambda_max = self.kkt_beta_zero(alpha)
        lambda_seq = self._get_lambda_seq(lambda_max, nlambda, ratio_lambda)
        # add the first solution (corresponds to lambda = lambda_max)
        
        beta_list = []
        niter_list = []
        tol_list = []
        for i in range(len(self.Xtlist)):
            
            pp = self.Xtlist[i].shape[0]
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
                w1, w2 = self._alpha_lambda_to_w1_w2(alpha, lam)
                betalist, niter_vec[idx], tol_vec[idx], (tlist, rlist, obj_lik, l1_beta, l2_beta) = self.solve_by_blk(
                    w1=w1, w2=w2, idx=i,
                    tol=tol, maxiter=maxiter,
                    init_beta=betalist, init_t=tlist, init_r=rlist, 
                    init_obj_lik=obj_lik, init_l1_beta=l1_beta, init_l2_beta=l2_beta,
                    XtX_diag_list=self.XtX_diag_list,
                    return_raw=True
                )
                beta_mat[:, idx] = self._merge_list(betalist)
            beta_list.append(beta_mat)
            niter_list.append(niter_vec)
            tol_list.append(tol_vec)
        beta_merged = np.concatenate(beta_list, axis=0)
        return beta_merged, lambda_seq, niter_vec, tol_vec
        
    def solve_path(self, alpha=0.5, tol=1e-5, maxiter=1000, nlambda=100, ratio_lambda=100):
        '''
        What it does:
            Solve for the regularization path which contains a sequence of lambda values.
            Determine lambda sequence from nlambda, ratio_lambda (lambda_max / lambda_min). 
            The lambda_max is determined from KKT condition at beta = 0.
        About lambda, alpha and w1, w2 conversion:
            w1 = lambda * alpha
            w2 = lambda * (1 - alpha) / 2
        '''
         
        # check input parameters
        self._solve_path_param_sanity_check(alpha, nlambda, ratio_lambda)
        
        # initialize the beta mat (p x nlambda)
        beta_mat = np.zeros((self.p, nlambda))
        # initialize niter and maxiter records
        niter_vec, tol_vec = np.zeros(nlambda), np.zeros(nlambda)
        # determine lambda sequence
        lambda_max = self.kkt_beta_zero(alpha)
        lambda_seq = self._get_lambda_seq(lambda_max, nlambda, ratio_lambda)
        # add the first solution (corresponds to lambda = lambda_max)
        beta_mat[:, 0] = np.zeros(self.p)
        # initialize beta, t, r
        betalist, tlist, rlist, obj_lik, l1_beta, l2_beta = None, None, None, None, None, None
        # loop over lambda sequence skipping the first, lambda_max
        for idx, lam in enumerate(lambda_seq):
            # print('working on idx = ', idx)
            if idx == 0:
                continue
            w1, w2 = self._alpha_lambda_to_w1_w2(alpha, lam)
            betalist, niter_vec[idx], tol_vec[idx], (tlist, rlist, obj_lik, l1_beta, l2_beta) = self.solve(
                w1=w1, w2=w2, tol=tol, maxiter=maxiter,
                init_beta=betalist, init_t=tlist, init_r=rlist, 
                init_obj_lik=obj_lik, init_l1_beta=l1_beta, init_l2_beta=l2_beta,
                XtX_diag_list=self.XtX_diag_list,
                return_raw=True
            )
            beta_mat[:, idx] = self._merge_list(betalist)
        return beta_mat, lambda_seq, niter_vec, tol_vec
        
        
    