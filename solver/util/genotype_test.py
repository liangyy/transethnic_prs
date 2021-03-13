import unittest
from parameterized import parameterized

import numpy as np

from solver.util.math import mean_center_col
from solver.util.genotype import *

class Test(unittest.TestCase):
    def setUp(self):
        # upper triangular
        self.geno_file_prefix = 'test_data/toy.chr1'
        self.geno_file_prefix2 = 'test_data/toy.chr2'
        self.geno_file_prefix_by_chr = 'test_data/toy.chr{chr_num}'
    def test_load_genotype_from_bedfile(self):
        geno, indiv, snp = load_genotype_from_bedfile(self.geno_file_prefix + '.bed')
        n, p = geno.shape
        self.assertEqual(n, len(indiv))
        self.assertEqual(p, snp.shape[0])
    def test_init(self):
        geno_io1 = GenotypeIO(self.geno_file_prefix)
        geno_io2 = GenotypeIO(self.geno_file_prefix_by_chr, chromosome_list=[1, 2])
        self.assertEqual(3, len(geno_io1.indiv_list))
        self.assertEqual(3, len(geno_io2.indiv_list))
    def test_calc_geno_mul_y(self):
        y = np.array([1.2232, -4.12, 0.221])
        y = mean_center_col(y)
        x, indiv, snp = load_genotype_from_bedfile(self.geno_file_prefix + '.bed')
        x = mean_center_col(x)
        Xty = x.T @ y
        geno_io1 = GenotypeIO(self.geno_file_prefix)
        geno_io2 = GenotypeIO(self.geno_file_prefix_by_chr, chromosome_list=[1, 2])
        df_y1 = pd.DataFrame({'indiv_id': geno_io1.indiv_list, 'y': y})
        df_y2 = pd.DataFrame({'indiv_id': geno_io2.indiv_list, 'y': y})
        kk1, snp1 = geno_io1.calc_geno_mul_y(df_y1)
        kk2, snp2 = geno_io2.calc_geno_mul_y(df_y2)
        np.testing.assert_allclose(kk1, kk2[:len(kk1)])
        np.testing.assert_allclose(kk1, Xty)
        self.assertEqual(snp1.shape[0], len(kk1))
        self.assertEqual(snp2.shape[0], len(kk2))
    def test_calc_geno_cov(self):
        x, indiv, snp = load_genotype_from_bedfile(self.geno_file_prefix + '.bed')  
        x = mean_center_col(x)  
        p = x.shape[1]
        geno_io1 = GenotypeIO(self.geno_file_prefix)
        geno_io2 = GenotypeIO(self.geno_file_prefix_by_chr, chromosome_list=[1, 2])
        covx = np.cov(x.T)
        cov1, snp1 = geno_io1.calc_geno_cov(mode='banded', params=10)
        cov2, snp2 = geno_io2.calc_geno_cov(mode='banded', params=10)
        np.testing.assert_allclose(np.triu(covx), cov1.toarray())
        np.testing.assert_allclose(np.triu(covx.T), cov1.toarray())
        
        np.testing.assert_allclose(cov1.toarray(), cov2.toarray()[:p, :p])
    @parameterized.expand([
        [ 0 ], 
        [ 1 ], 
        [ 2 ], 
        [ 3 ], 
        [ 4 ], 
        [ 5 ] 
    ])
    def test_calc_geno_cov_check_band(self, offset):
        x, indiv, snp = load_genotype_from_bedfile(self.geno_file_prefix2 + '.bed') 
        x = mean_center_col(x)  
        p = x.shape[1]
        covx = np.cov(x.T)
        geno_io1 = GenotypeIO(self.geno_file_prefix2)
        cov1_2, snp1_2 = geno_io1.calc_geno_cov(mode='banded', params=offset - 1)
        band_mask = np.tri(p, p, dtype=bool)&~np.tri(p, p, -offset, dtype=bool)
        tmp = covx.copy()
        tmp[~band_mask.T] = 0
        np.testing.assert_allclose(np.triu(tmp), cov1_2.toarray())
    @parameterized.expand([
        [ 0.1 ], 
        [ 0.2 ], 
        [ 0.4 ], 
        [ 0.5 ], 
        [ 1.0 ],
        [ 1.1 ],
    ])
    def test_calc_geno_cov_check_band(self, thres):
        x, indiv, snp = load_genotype_from_bedfile(self.geno_file_prefix2 + '.bed') 
        x = mean_center_col(x)  
        p = x.shape[1]
        covx = np.cov(x.T)
        geno_io1 = GenotypeIO(self.geno_file_prefix2)
        cov1_2, snp1_2 = geno_io1.calc_geno_cov(mode='cap', params=thres)
        band_mask = np.absolute(covx) > thres
        tmp = covx.copy()
        tmp[~band_mask] = 0
        np.testing.assert_allclose(np.triu(tmp), cov1_2.toarray())
    
        
if __name__ == '__main__':
    unittest.main()
