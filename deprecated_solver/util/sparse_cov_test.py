import unittest
from parameterized import parameterized

import numpy as np
from scipy.sparse import save_npz, csr_matrix
from solver.util.sparse_cov import SparseCov

class Test(unittest.TestCase):
    def setUp(self):
        # upper triangular
        row = np.array([0, 0, 1, 1, 2, 2])
        col = np.array([0, 2, 2, 3, 4, 5])
        data = np.array([1.2, 2.2, 3.5, 4.2, 5.1, -6.9])
        self.mat_csr = csr_matrix((data, (row, col)), shape=(6, 6))
        self.file_csr = 'test_data/mat_csr.sparse_cov_test'
        save_npz(self.file_csr, self.mat_csr)
        self.mat_dense = np.triu(self.mat_csr.toarray()) + np.triu(self.mat_csr.toarray(), k=1).T
        self.vec = np.array([0.1, 3.2, -3.09, 3.2, -9.887, 3.001])

        # not upper triangular
        row = np.array([0, 4, 5, 4, 0, 1])
        col = np.array([0, 2, 0, 3, 4, 0])
        data = np.array([1.2, 2.2, 3.5, 4.2, 5.1, -6.9])
        self.mat_csr2 = csr_matrix((data, (row, col)), shape=(6, 6))
        self.file_csr2 = 'test_data/mat_csr2.sparse_cov_test'
        save_npz(self.file_csr2, self.mat_csr2)
        self.mat_dense2 = np.triu(self.mat_csr2.toarray()) + np.triu(self.mat_csr2.toarray(), k=1).T
    def test_init_from_mat(self):
        A = SparseCov(matrix=self.mat_csr)
        A2 = SparseCov(matrix=self.mat_csr2)
        np.testing.assert_allclose(A.csr.toarray(), np.triu(self.mat_dense))   
        np.testing.assert_allclose(A2.csr.toarray(), np.triu(self.mat_dense2))   
    def test_init_from_file(self):
        A = SparseCov(filename=self.file_csr + '.npz')
        A2 = SparseCov(filename=self.file_csr2 + '.npz')
        np.testing.assert_allclose(A.csr.toarray(), np.triu(self.mat_dense))   
        np.testing.assert_allclose(A2.csr.toarray(), np.triu(self.mat_dense2)) 
    def test_get_diag_as_vec(self):
        A = SparseCov(matrix=self.mat_csr)
        A2 = SparseCov(matrix=self.mat_csr2)
        np.testing.assert_allclose(A.get_diag_as_vec(), self.mat_dense.diagonal())   
        np.testing.assert_allclose(A2.get_diag_as_vec(), self.mat_dense2.diagonal())  
    def test_mul_vec(self):
        A = SparseCov(matrix=self.mat_csr)
        A2 = SparseCov(matrix=self.mat_csr2)
        np.testing.assert_allclose(A.mul_vec(self.vec), self.mat_dense @ self.vec)   
        np.testing.assert_allclose(A2.mul_vec(self.vec), self.mat_dense2 @ self.vec)  
    @parameterized.expand([
        [ 0 ], 
        [ 1 ], 
        [ 2 ], 
        [ 3 ], 
        [ 4 ], 
        [ 5 ] 
    ])
    def test_get_row_as_vec(self, idx):
        A = SparseCov(matrix=self.mat_csr)
        A2 = SparseCov(matrix=self.mat_csr2)
        np.testing.assert_allclose(A.get_row_as_vec(idx), self.mat_dense[idx, :])  
        np.testing.assert_allclose(A2.get_row_as_vec(idx), self.mat_dense2[idx, :])  
    @parameterized.expand([
        [0.1, 1.2], [1.0, 0.1], [-2, -3], [-3, 1.2]
    ])
    def test_add(self, coef1, coef2):
        A = SparseCov(matrix=self.mat_csr)
        A2 = SparseCov(matrix=self.mat_csr2)
        np.testing.assert_allclose(
            A.add(A2, coef1=coef1, coef2=coef2).csr.toarray(),
            np.triu(coef1 * self.mat_dense + coef2 * self.mat_dense2)
        )    
        np.testing.assert_allclose(
            A2.add(A, coef1=coef2, coef2=coef1).csr.toarray(),
            np.triu(coef1 * self.mat_dense + coef2 * self.mat_dense2)
        )    

if __name__ == '__main__':
    unittest.main()
