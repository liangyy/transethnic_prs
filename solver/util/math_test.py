import unittest

import numpy as np

from solver.util.math import *


class TestSoftThres(unittest.TestCase):
    def setUp(self):
        self.mat = np.array([
            [0, 1, 3.2],
            [-2.3, 23.1, 30],
            [2.3, -33.1, -33.2],
            [0.0, 0.0, 0.0]
        ])
    def test_int_1(self):
        with self.assertRaises(TypeError):
            result = soft_thres(1, 2.0)
    def test_int_2(self):
        with self.assertRaises(TypeError):
            result = soft_thres(1.0, 2)
    def test_int_12(self):
        with self.assertRaises(TypeError):
            result = soft_thres(1, 2)
    def test_neg(self):
        with self.assertRaises(TypeError):
            result = soft_thres(1.0, -2.0)
    def test_return_neg(self):
        self.assertAlmostEqual(soft_thres(-0.7, 0.5), -0.2)
    def test_return_pos(self):
        self.assertAlmostEqual(soft_thres(0.7, 0.5), 0.2)
    def test_return_zero(self):
        self.assertAlmostEqual(soft_thres(-0.2, 0.5), 0.0)
    def test_return_zero(self):
        self.assertAlmostEqual(soft_thres(-0.2, 0.5), 0.0)
    def test_mean_center_col(self):
        tmp = np.zeros(self.mat.shape)
        for i in range(tmp.shape[1]):
            tmp[:, i] = self.mat[:, i] - np.mean(self.mat[:, i])
        np.testing.assert_allclose(mean_center_col(self.mat), tmp)
if __name__ == '__main__':
    unittest.main()
    