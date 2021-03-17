# import unittest
# 
# import numpy as np
# import scipy.sparse as spr
# from solver.util.sparse_mat import *
# 
# class Test(unittest.TestCase):
#     def setUp(self):
#         row = np.array([0, 0, 1, 2, 2, 2])
#         col = np.array([0, 2, 2, 0, 1, 2])
#         data = np.array([1.2, 2.2, 3.5, 4.2, 5.1, -6.9])
#         self.mat_csr = spr.csr_matrix((data, (row, col)), shape=(3, 3))
#         self.mat_dense = np.array([[1.2, 0, 2.2], [0, 0, 3.5], [4.2, 5.1, -6.9]])
#         self.vec = np.array([0.1, 3.2, -3.09])
#     def test_get_diag_as_vec(self):
#         np.testing.assert_allclose(get_diag_as_vec(self.mat_csr), np.diagonal(self.mat_dense))    
#     def test_mul_vec(self):
#         np.testing.assert_allclose(mul_vec(self.mat_csr, self.vec), self.mat_dense @ self.vec)    
#     def test_get_row_as_vec(self):
#         idx = 2
#         np.testing.assert_allclose(get_row_as_vec(self.mat_csr, idx), self.mat_dense[idx, :])   
# 
# if __name__ == '__main__':
#     unittest.main()
# 