import unittest
from parameterized import parameterized

import numpy as np
import scipy.sparse as spr
from sklearn.linear_model import ElasticNet
from solver.model1 import Model1
from solver.util.math import soft_thres, mean_center_col
# from solver.util.sparse_mat import get_diag_as_vec
from solver.util.sparse_cov import SparseCov
from solver.util.objective import obj_original, obj_model1

class TestModel1(unittest.TestCase):
    def setUp(self):
        row1 = np.array([0, 1, 2, 0, 2, 1, 0, 0, 0])
        col1 = np.array([0, 1, 2, 2, 0, 0, 1, 3, 4])
        row2 = np.array([0, 1, 2, 0, 1, 0, 0])
        col2 = np.array([0, 1, 2, 1, 0, 3, 4])
        data1 = np.array([10, 5, 10, 2.2, 2.2, -1.2, -1.2, -2, 1])
        data2 = np.array([8, 7, 10, 1.0, 1.0, -.3, 2])
        self.genofile_X1 = spr.csr_matrix(self.mean_center(spr.csr_matrix((data1, (row1, col1)), shape=(3, 5))))
        self.Rhat1, covx = self.geno2cov(self.genofile_X1.toarray())
        self.Rhat1 = SparseCov(matrix=spr.triu(self.Rhat1))
        self.y1 = self.mean_center(np.array([1.4, 3.3, -1.7]))
        self.Xty1 = self.genofile_X1.toarray().T @ self.y1
        self.bhat1 = self.get_bhat(self.genofile_X1.toarray(), self.y1)
        self.N1 = self.genofile_X1.shape[0]
        self.genofile_X2 = spr.csr_matrix(self.mean_center(spr.csr_matrix((data2, (row2, col2)), shape=(3, 5))))
        self.y2 = self.mean_center(np.array([10.3, -5, 2.3]))
        self.Xty2 = self.genofile_X2.toarray().T @ self.y2
        self.XtX2 = SparseCov(matrix=spr.triu(self.genofile_X2.T @ self.genofile_X2))
        self.p = self.genofile_X1.shape[1]
    def mean_center(self, mat):
        return mat - mat.mean(axis=0)
    def get_bhat(self, x, y):
        y_ = y - y.mean()
        x_ = x - x.mean(axis=0)
        return x_.T @ y_ / np.diagonal(x_.T @ x_)
    def geno2cov(self, geno_x):
        covx = np.cov(geno_x.T)
        return spr.csr_matrix(covx), covx

    def test_check_bhat(self):
        np.testing.assert_allclose(
            self.Xty1, 
            (self.N1 - 1) * self.Rhat1.get_diag_as_vec() * self.bhat1
        )
    def test_init(self):
        mod = Model1(
            self.Rhat1, self.bhat1, self.N1, 
            self.XtX2, self.Xty2
        )
        X2 = self.genofile_X2.toarray()
        A = (self.N1 - 1) * self.Rhat1.to_mat().toarray() + X2.T @ X2
        b = (self.N1 - 1) * np.diagonal(self.Rhat1.to_mat().toarray()) * self.bhat1 + X2.T @ self.y2
        np.testing.assert_allclose(mod.A.to_mat().toarray(), A)
        np.testing.assert_allclose(mod.b, b)
    def test_solve_one_step(self):
        w1 = 0.3
        w2 = 0.1
        mod = Model1(
            self.Rhat1, self.bhat1, self.N1, 
            self.XtX2, self.Xty2
        )
        beta, _, _ = mod.solve(w1, w2, maxiter=1)

        # manual iteration
        A = mod.A.to_mat().toarray()
        b = mod.b
        a = 2 * w2 + 2 * np.diagonal(A)
        beta_man = np.zeros(A.shape[0])
        t = np.zeros(A.shape[0])
        for j in range(t.shape[0]):
            a_ = a[j]
            c_ = 2 * t[j] - A[j, j] * beta_man[j] - 2 * b[j]
            beta_new = - soft_thres(c_, w1) / a_
            t = t - A[:, j] * (beta_man[j] - beta_new)
            beta_man[j] = beta_new
        np.testing.assert_allclose(beta_man, beta)
    @parameterized.expand([
        [0.1, 0.3, 10, 1e-5],
        [0.1, 0.3, 100, 1e-5],
        [0.1, 0.3, 1000, 1e-5],
        [0.1, 0.3, 10000, 1e-5],
        [3., 1., 10000, 1e-1],
        [3., 1., 10000, 1e-2],
        [3., 1., 10000, 1e-3],
        [3., 1., 10000, 1e-4],
    ])
    def test_solve2(self, w1, w2, maxiter, tol):
        mod = Model1(
            self.Rhat1, self.bhat1, self.N1, 
            self.XtX2, self.Xty2
        )    
        beta, niter, diff = mod.solve(w1=w1, w2=w2, tol=tol, maxiter=maxiter)
        self.assertEqual(beta.shape[0], self.p)
        self.assertTrue(niter == maxiter or diff <= tol)
    @parameterized.expand([
        [0.1, 10, 10],
        [0.1, 100, 10],
        [0.5, 10, 100],
        [0.5, 100, 100]
    ])
    def test_solve_path(self, alpha, nlambda, ratio_lambda):
        mod = Model1(
            self.Rhat1, self.bhat1, self.N1, 
            self.XtX2, self.Xty2
        )  
        beta, lam_seq = mod.solve_path(alpha=alpha, nlambda=nlambda, ratio_lambda=ratio_lambda)
        self.assertEqual(beta.shape[0], self.p)
        self.assertEqual(beta.shape[1], nlambda)
        self.assertAlmostEqual(lam_seq[0] / lam_seq[-1], ratio_lambda)
    @parameterized.expand([
        [10.0, 0.0],
        [10.0, 0.9],
        [10.0, 0.5],
        [0.1, 0.0],
        [0.1, 0.9],
        [0.1, 0.5],
    ])
    def test_solve_cmp_w_sklearn(self, alpha, l1_ratio):
        maxiter = 1000
        tol = 1e-4

        # sklearn EN
        X = np.concatenate([self.genofile_X1.toarray(), self.genofile_X2.toarray()], axis=0)
        y = np.concatenate([self.y1, self.y2], axis=0)
        X = mean_center_col(X)
        y = mean_center_col(y)
        regr = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, copy_X=True,
            fit_intercept=False, max_iter=maxiter, tol=tol, selection='cyclic'
        )
        regr.fit(X, y)
        beta_sk = regr.coef_

        # my solver
        w1 = alpha * l1_ratio * 2 * X.shape[0]
        w2 = 0.5 * alpha * (1 - l1_ratio) * 2 * X.shape[0]
        mod = Model1(
            self.Rhat1, self.bhat1, self.N1, 
            self.XtX2, self.Xty2
        ) 
        beta_my, _, _ = mod.solve(w1=w1, w2=w2, maxiter=maxiter, tol=tol)
        np.testing.assert_allclose(obj_original(beta_sk, X, y, w1, w2), obj_original(beta_my, X, y, w1, w2))
        np.testing.assert_allclose(obj_model1(beta_sk, mod.A.to_mat(), mod.b, w1, w2), obj_model1(beta_my, mod.A.to_mat(), mod.b, w1, w2))



if __name__ == '__main__':
    unittest.main()
