import unittest

from solver.util.math import soft_thres


class TestSoftThres(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()