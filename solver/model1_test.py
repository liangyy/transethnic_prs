import unittest
from parameterized import parameterized

import numpy as np
import scipy.sparse as spr
from solver.model1 import Model1
from solver.util.math import soft_thres

class TestModel1(unittest.TestCase):
    def setUp(self):
        row1 = np.array([0, 1, 2, 0, 2, 1, 0])
        col1 = np.array([0, 1, 2, 2, 0, 0, 1])
        row2 = np.array([0, 1, 2, 0, 1])
        col2 = np.array([0, 1, 2, 1, 0])
        data1 = np.array([10, 5, 10, 2.2, 2.2, -1.2, -1.2])
        data2 = np.array([8, 7, 10, 1.0, 1.0])
        self.Rhat1 = spr.csr_matrix((data1, (row1, col1)), shape=(3, 3))
        self.bhat1 = np.array([0.2, 0.1, -4.0])
        self.N1 = 10
        self.genofile_X2 = spr.csr_matrix((data2, (row2, col2)), shape=(3, 3))
        self.y2 = np.array([10.3, -5, 2.3])
    def test_init(self):
        mod = Model1(
            self.Rhat1, self.bhat1, self.N1, 
            self.genofile_X2, self.y2
        )
        X2 = self.genofile_X2.toarray()
        A = (self.N1 - 1) * self.Rhat1.toarray() + X2.T @ X2
        b = (self.N1 - 1) * np.diagonal(self.Rhat1.toarray()) * self.bhat1 + X2 @ self.y2
        np.testing.assert_allclose(mod.A.toarray(), A)
        np.testing.assert_allclose(mod.b, b)
    def test_solve_one_step(self):
        w1 = 0.3
        w2 = 0.1
        mod = Model1(
            self.Rhat1, self.bhat1, self.N1, 
            self.genofile_X2, self.y2
        )
        beta, _, _ = mod.solve(w1, w2, maxiter=1)
        
        # manual iteration
        A = mod.A.toarray()
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
        [0.1, 0.3, 10, 1e-5, 3],
        [0.1, 0.3, 100, 1e-5, 3],
        [0.1, 0.3, 1000, 1e-5, 3],
        [0.1, 0.3, 10000, 1e-5, 3],
        [3., 1., 10000, 1e-1, 3],
        [3., 1., 10000, 1e-2, 3],
        [3., 1., 10000, 1e-3, 3],
        [3., 1., 10000, 1e-4, 3],
    ])
    def test_solve2(self, w1, w2, maxiter, tol, out_shape):
        mod = Model1(
            self.Rhat1, self.bhat1, self.N1, 
            self.genofile_X2, self.y2
        )    
        beta, niter, diff = mod.solve(w1=w1, w2=w2, tol=tol, maxiter=maxiter)
        self.assertEqual(beta.shape[0], out_shape)
        self.assertTrue(niter == maxiter or diff <= tol)
    

if __name__ == '__main__':
    unittest.main()
    