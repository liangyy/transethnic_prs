import numpy as np
from tqdm import tqdm

from transethnic_prs.util.math import diag_mul_mat, mat_mul_diag
from transethnic_prs.util.misc import check_np_darray
from transethnic_prs.util.linear_model import fast_simple_linear_regression

class DataSimulator:
    def __init__(self, X, beta, VarX=None, no_chol=False, disable_progress_bar=False):
        nx, pxs = self._set_x(X)
        self._set_beta(beta, pxs)
        self.pxs = pxs
        self.nx = nx
        self._set_varx(VarX)
        if no_chol is False:
            self._calc_chol(disable_progress_bar=disable_progress_bar)
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
    def _calc_chol(self, thres=1e-10, disable_progress_bar=True):
        chol_varx = []
        for varx in tqdm(self.VarX, disable=disable_progress_bar):
            w, v = np.linalg.eigh(varx)
            v = v[:, np.abs(w) > thres]
            w = w[np.abs(w) > thres]
            if (w < 0).sum() > 0:
                raise ValueError('Not positive semi-definite.')
            v = mat_mul_diag(v, np.sqrt(w))
            chol_varx.append(v)
        self.chol_varx = chol_varx
    def update_beta(self, beta):
        self._set_beta(beta, self.pxs)
        self._calc_sigma2g()
    def sim_gwas(self, h2, sample_size):
        vary = self.sigma2g / h2
        prefactor = np.sqrt(vary / sample_size)
        s_vec = [ np.sqrt(vary / (sample_size * varx.diagonal())) for varx in self.VarX ]
        betahat = []
        se = []
        for varx, s_vec_blk, beta, lx in zip(self.VarX, s_vec, self.beta, self.chol_varx):
            sx = np.sqrt(varx.diagonal())
            mu = diag_mul_mat(s_vec_blk / sx, varx) @ (beta / s_vec_blk / sx)
            z = np.random.normal(size=lx.shape[1])
            bhat = mu + prefactor / (sx ** 2) * (lx @ z)
            betahat.append(bhat)
            se.append(prefactor / sx)
        return betahat, se
    def sim_x_and_y(self, h2, sample_size):
        xlist = self._sim_x(sample_size)
        return xlist, self._sim_y(h2, xlist)
    def sim_y(self, h2):
        return self._sim_y(h2, self.X)
    def _sim_y(self, h2, xlist):
        sigma2e = self.sigma2g / h2 * (1 - h2)
        mu = np.zeros(xlist[0].shape[0])
        for x, beta in zip(xlist, self.beta):
            mu += x @ beta
        return np.random.normal(loc=mu, scale=np.sqrt(sigma2e))
    def _sim_x(self, sample_size):
        xlist = []
        for x, lx in zip(self.X, self.chol_varx):
            p, k = lx.shape
            z = np.random.normal(size=(sample_size, k))
            mu_x = np.mean(x, axis=0)
            xlist.append(mu_x[np.newaxis, :] + z @ lx.T)
        return xlist
    def calc_gwas_from_y(self, y):
        bhat = []
        se = []
        for x in self.X:
            bhat_, se_ = fast_simple_linear_regression(x, y)
            bhat.append(bhat_)
            se.append(se_)
        return bhat, se
