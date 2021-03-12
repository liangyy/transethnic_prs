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
import solver.util.sparse_mat as spr
import solver.util.math as math

class Model1:
    def __init__(self, Rhat1, bhat1, N1, genofile_X2, y2):
        XtX2, Xty2 = geno.load_genofile_as_XtX(genofile_X2, y2)
        self.A = (N1 - 1) * Rhat1 + XtX2
        self.b = (N1 - 1) * spr.get_diag_as_vec(Rhat1) * bhat1 + Xty2
        # p: number of predictors
        self.p = self.A.shape[0]
    def solve(self, w1, w2, tol=1e-5, maxiter=1000):
        beta = self._init_beta()
        tvec = spr.mul_vec(self.A, beta)
        avec = 2 * w2 + 2 * spr.get_diag_as_vec(self.A)
        diff, niter = np.Inf, 0
        while diff > tol and niter < maxiter:
            diff_ = 0
            for j in range(self.p):
                a_ = avec[j]
                c_ = 2 * (tvec[j] - self.A[j, j] * beta[j]) - 2 * self.b[j]
                beta_j_new = - math.soft_thres(c_, w1) / a_
                tvec = tvec - spr.get_row_as_vec(self.A, j) * (beta[j] - beta_j_new)
                diff_ += (beta_j_new - beta[j]) ** 2
                beta[j] = beta_j_new
            diff = np.sqrt(diff_)
            niter += 1
        return beta, niter, diff
    def _init_beta(self):
        return np.zeros(self.p)
    
    