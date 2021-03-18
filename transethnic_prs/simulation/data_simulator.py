import numpy as np

from transethnic_prs.util.math import diag_mul_mat, mat_mul_diag
from transethnic_prs.util.misc import check_np_darray
from transethnic_prs.util.linear_model import fast_simple_linear_regression

class DataSimulator:
    def __init__(self, X, beta, VarX=None):
        nx, pxs = self._set_x(X)
        self._set_beta(beta, pxs)
        self.nx = nx
        self._set_varx(VarX)
        self._calc_chol()
        self._calc_sigma2g()
    def _set_x(self, X):
        pxs = []
        nx = None
        for x in X:
            nx_, px_ = check_np_darray(x, dim=2)
            if nx is not None and nx_ != nx:
                raise ValueError('Elements in X have different number of samples.')
            elif nx is None:
                nx = nx_
            pxs.append(px_)
        self.X = X
        return nx, pxs
    def _set_beta(self, beta, pxs):
        '''
        if beta is a list, will merge and treat as 1-dim np.ndarray
        '''
        if isinstance(beta, list):
            beta_ = np.concatenate(beta, axis=0)
        else:
            beta_ = beta
        pb = check_np_darray(beta_)
        if pb != np.array(pxs).sum():
            raise ValueError('Different number of SNPs in beta and X.')
        offset = 0
        self.beta = []
        for p in pxs:
            self.beta.append(beta_[offset : (offset + p)])
            offset += p
    def _set_varx(self, VarX):
        if VarX is None:
            self.VarX = [ np.cov(x.T) for x in self.X ]
        else:
            for varx, x in zip(VarX, self.X):
                p = check_np_darray(varx, dim=2, check_squared=True)
                if p != x.shape[1]:
                    raise ValueError('Different number of SNPs in VarX and X.')
        self.VarX = VarX
    def _calc_sigma2g(self):
        sigma2g = 0
        for beta, varx in zip(self.beta, self.VarX):
            sigma2g += beta.T @ (varx @ beta)
        self.sigma2g = sigma2g 
    def _calc_chol(self, thres=1e-10):
        chol_varx = []
        for varx in self.VarX:
            w, v = np.linalg.eigh(varx)
            v = v[:, np.abs(w) > thres]
            w = w[np.abs(w) > thres]
            v = mat_mul_diag(v, w)
            chol_varx.append(v)
        self.chol_varx = chol_varx
    def sim_gwas(self, h2, sample_size):
        vary = self.sigma2g / h2
        prefactor = np.sqrt(vary) / sample_size
        s_vec = [ np.sqrt(vary / (sample_size ** 2 * varx.diagonal())) for varx in self.VarX ]
        betahat = []
        for varx, s_vec_blk, beta, lx in zip(self.VarX, s_vec, self.beta, self.chol_varx):
            sx = np.sqrt(varx.diagonal())
            mu = diag_mul_mat(s_vec_blk / sx, varx) @ (beta / s_vec_blk / sx)
            z = np.random.normal(size=lx.shape[1])
            bhat = mu + prefactor * (s_vec_blk ** 2) * (lx @ z)
            betahat.append(bhat)
        return betahat
    def sim_y(self, h2):
        sigma2e = self.sigma2g / h2 * (1 - h2)
        mu = np.zeros(self.nx)
        for x, beta in zip(self.X, self.beta):
            mu += x @ beta
        return np.random.normal(loc=mu, scale=np.sqrt(sigma2e))
    def calc_gwas_from_y(self, y):
        bhat = []
        se = []
        for x in self.X:
            bhat_, se_ = fast_simple_linear_regression(x, y)
            bhat.append(bhat_)
            se.append(se_)
        return bhat, se
