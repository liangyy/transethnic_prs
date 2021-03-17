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

class Model1Blk:
    '''
    We take specific data structure for A and b.
    We assume A is block-wise diagonal.
    Alist = [ A1, ..., Ak ]
    blist = [ b1, ..., bk ]
    , where Ai is the ith squared matrix along the diagonal of A, 
    i.e. Ai = Xi'Xi
    and bi is the corresponding Xi'y
    '''
    def __init__(self, Alist, blist, X, y):
        '''
        self.Alist
        self.blist
        self.X
        self.y
        '''
        self._set_a_and_b(Alist, blist)
        self._set_x_and_y(X, y)
    def _set_a_and_b(self, Alist, blist):
        if len(Alist) != len(blist):
            raise ValueError('Alist and blist have different length.')
        for i in range(len(Alist)):
            na, _ = check_np_darray(Alist[i], dim=2, check_squared=True)
            nb = check_np_darray(blist[i], dim=1)
            if na != nb:
                raise ValueError(f'The {i}th element in Alist and blist has un-matched shape {na} != {nb}.')
        self.Alist = Alist
        self.blist = blist
    def _set_x_and_y(self, x, y):
        nx, _ = check_np_darray(x, dim=2)
        ny = check_np_darray(y, dim=1)
        if nx != ny:
            raise ValueError(f'X.shape[0] != y.shape[0].') 
        self.X = x
        self.y = y       
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
        for b in self.blist:
            lambda_max = max(
                lambda_max, 
                2 * np.absolute(b).max() / alpha
            )
        return lambda_max
    def solve(self, w1, w2, tol=1e-5, maxiter=1000):
        betalist, niter, diff = solve_by_dense_blk(
            self.Alist, self.blist, self.X, self.y, 
            w1=w1, w2=w2, tol=tol, maxiter=maxiter
        )
        beta = np.concatenate(betalist, axis=0)
        return beta, niter, diff
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
        pass 
        # check input parameters
        self._solve_path_param_sanity_check(alpha, nlambda, ratio_lambda)
        
        # initialize the beta mat (p x nlambda)
        beta_mat = np.zeros((self.p, nlambda))
        # initialize niter and maxiter records
        niter_vec, maxiter_vec = np.zeros(nlambda), np.zeros(nlambda)
        # determine lambda sequence
        lambda_max = self.kkt_beta_zero(alpha)
        lambda_seq = self._get_lambda_seq(lambda_max, nlambda, ratio_lambda)
        # add the first solution (corresponds to lambda = lambda_max)
        beta_mat[:, 0] = np.zeros(self.p)
        # loop over lambda sequence skipping the first, lambda_max
        for idx, lam in enumerate(lambda_seq):
            if idx == 0:
                continue
            w1, w2 = self._alpha_lambda_to_w1_w2(alpha, lam)
            beta_mat[:, idx], niter_vec[idx], maxiter_vec[idx] = self.solve(w1=w1, w2=w2, tol=tol, maxiter=maxiter)
        return beta_mat, lambda_seq, niter_vec, maxiter_vec
        
        
    