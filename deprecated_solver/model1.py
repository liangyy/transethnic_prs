'''

Model1 for trans-ethnic PRS.

Model/loss:
  || y1 - X1 * beta ||_2^2 +  || y2 - X2 * beta ||_2^2
  
Penalization: 
  w1 || beta ||_1 + w2 || beta ||_2^2

Where: 
  X1 and y1 are not available. 
  Instead, we know 
    X1'X1 = (N1 - 1) * Rhat1
    X1'y1 = (N1 - 1) * diag(Rhat1) * bhat1
    
So, the problem being solved is actually:
  argmin_beta beta' * A * beta - 2 * b' * beta + w1 || beta ||_1 + w2 || beta ||_2^2
, where
  A = (N1 - 1) Rhat1 + X2'X2
  b = (N1 - 1) diag(Rhat1) bhat1 + X2'y2

'''
import numpy as np

import solver.util.genotype as geno
# import solver.util.sparse_mat as spr
import solver.util.math as math

class Model1:
    def __init__(self, Rhat1, bhat1, N1, XtX2, Xty2):
        # XtX2, Xty2 = math.calc_XtX_and_Xty(X2, y2)
        '''
        Rhat1 and XtX2 are assumed to be class SparseCov
        '''
        self.A = Rhat1.add(XtX2, coef1=N1 - 1, coef2=1)
        self.b = (N1 - 1) * Rhat1.get_diag_as_vec() * bhat1 + Xty2
        # p: number of predictors
        self.p = self.A.size
    def _init_beta(self):
        return np.zeros(self.p)
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
        return 2 * np.absolute(self.b).max() / alpha
    def solve(self, w1, w2, tol=1e-5, maxiter=1000):
        beta = self._init_beta()
        tvec = self.A.mul_vec(beta)
        avec = 2 * w2 + 2 * self.A.get_diag_as_vec()
        diff, niter = np.Inf, 0
        while diff > tol and niter < maxiter:
            diff_ = 0
            for j in range(self.p):
                a_ = avec[j]
                c_ = 2 * (tvec[j] - self.A.get_jth_diag(j) * beta[j]) - 2 * self.b[j]
                beta_j_new = - math.soft_thres(c_, w1) / a_
                tvec = tvec - self.A.get_row_as_vec(j) * (beta[j] - beta_j_new)
                diff_ += (beta_j_new - beta[j]) ** 2
                beta[j] = beta_j_new
            diff = np.sqrt(diff_)
            niter += 1
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
        
        # check input parameters
        self._solve_path_param_sanity_check(alpha, nlambda, ratio_lambda)
        
        # initialize the beta mat (p x nlambda)
        beta_mat = np.zeros((self.p, nlambda))
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
            beta_mat[:, idx], _, _ = self.solve(w1=w1, w2=w2, tol=tol, maxiter=maxiter)
        return beta_mat, lambda_seq
        
        
    