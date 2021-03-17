import warnings

import re
from os.path import exists
import scipy.sparse 
import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin 


from solver.util.misc import list_equal, merge_two_lists
from solver.util.math import mean_center_col
from solver.util.sparse_cov import CovConstructor

CHR_WILDCARD = '{chr_num}'

def _load_fam(fn):
    tmp = pd.read_csv(fn, sep='\s+', header=None)
    indiv_list = merge_two_lists(list(tmp.iloc[:, 0]), list(tmp.iloc[:, 1]))
    return indiv_list

def load_genotype_from_bedfile(bedfile, snplist_to_exclude=None, chromosome=None, 
    missing_rate_cutoff=0.5, return_snp=False, standardize=False):
    
    G = read_plink1_bin(bedfile, verbose=False)
    
    if chromosome is not None:
        chr_str = G.chrom[0].values.tolist()
        if 'chr' in chr_str:
            chromosome = 'chr' + str(chromosome)
        else:
            chromosome = str(chromosome)
        G = G.where(G.chrom == chromosome, drop=True)
    
    snpid = G.variant.snp.to_series().values
    # snpid = np.array([ s.split('_')[1] for s in snpid ])
    if return_snp is True:
        a0 = G.variant.a0.to_series().to_numpy()
        a1 = G.variant.a1.to_series().to_numpy()       
        chrom = G.variant.chrom.to_series().to_numpy()    
    
    # load indiv list
    indiv_list = merge_two_lists(G.fid.to_series().tolist(), G.iid.to_series().tolist())
    
    geno = G.sel().values
    
    # filter out unwanted snps
    if snplist_to_exclude is not None:
        geno = geno[:, ~np.isin(snpid, snplist_to_exclude)]
        if return_snp is True:
            a0 = a0[~np.isin(snpid, snplist_to_exclude)]
            a1 = a1[~np.isin(snpid, snplist_to_exclude)]
            chrom = chrom[~np.isin(snpid, snplist_to_exclude)]
        
        snpid = snpid[~np.isin(snpid, snplist_to_exclude)]
   
    # filter out genotypes with high missing rate
    missing_rate = np.isnan(geno).mean(axis=0)
    geno = geno[:, missing_rate < missing_rate_cutoff]
    if return_snp is True:
        snpid = snpid[missing_rate < missing_rate_cutoff]
        a0 = a0[missing_rate < missing_rate_cutoff]
        a1 = a1[missing_rate < missing_rate_cutoff]
        chrom = chrom[missing_rate < missing_rate_cutoff]
        
    maf = np.nanmean(geno, axis=0) / 2
    
    # impute genotype missing value
    miss_x, miss_y = np.where(np.isnan(geno))
    geno[(miss_x, miss_y)] = maf[miss_y] * 2
    var_geno = 2 * maf * (1 - maf)
    
    # keep only genotypes with variance != 0
    to_keep = var_geno != 0
    geno = geno[:, to_keep]
    if return_snp is True:
        snpid = snpid[to_keep]
        a0 = a0[to_keep]
        a1 = a1[to_keep]
        chrom = chrom[to_keep]
        
    maf = maf[to_keep]
    var_geno = var_geno[to_keep]
    if standardize is True:
        geno = (geno - 2 * maf) / np.sqrt(var_geno)
    
    if return_snp is True:
        return geno, indiv_list, np.sqrt(var_geno), pd.DataFrame({'snpid': snpid.tolist(), 'a0': a0.tolist(), 'a1': a1.tolist(), 'chrom': chrom.tolist()})
    else:
        return geno, indiv_list, np.sqrt(var_geno)

        

class GenotypeIO:
    def __init__(self, bed_file_pattern, chromosome_list=None, snp_to_exclude=None, missing_rate_cutoff=0.5):
        self.file_prefix = bed_file_pattern
        self.snp_to_exclude = snp_to_exclude
        self.missing_rate_cutoff = missing_rate_cutoff
        self.chromosome_list = chromosome_list
        if CHR_WILDCARD in bed_file_pattern:
            if self.chromosome_list is None:
                # use 1 .. 22 chromosomes if chromosome_list is not specified
                self.chromosome_list = [ i for i in range(1, 23) ]
            self._check_all_bed_files_exist()
            self.read_by_chromosome = True
        else:
            self.read_by_chromosome = False
        self._get_indiv_list()
    def _check_all_bed_files_exist(self):
        for i in self.chromosome_list:
            fn1 = re.sub(CHR_WILDCARD, str(i), self.file_prefix) + '.bed'
            fn2 = re.sub(CHR_WILDCARD, str(i), self.file_prefix) + '.bim'
            if exists(fn1) and exists(fn2):
                continue
            else:
                raise ValueError(f'{fn1} and/or {fn2} do not exist.')
    def _get_indiv_list(self):
        if self.read_by_chromosome is False:
            self.indiv_list = _load_fam(self.file_prefix + '.fam')
        else:
            indiv_list = None
            for i in self.chromosome_list:
                file_prefix_i = re.sub(CHR_WILDCARD, str(i), self.file_prefix)
                indiv_list_i = _load_fam(file_prefix_i + '.fam')
                if indiv_list is None or list_equal(indiv_list, indiv_list_i):
                    indiv_list = indiv_list_i
                else:
                    raise ValueError('For multiple chromosomes, we need all individual lists are exactly the same, including the order.')
            self.indiv_list = indiv_list
    def calc_geno_mul_y(self, df_y):
        if self.read_by_chromosome is False:
            geno_mul_y, snp_meta = self._calc_geno_mul_y(self.file_prefix, df_y)
        else:
            geno_mul_y_list = []
            snp_meta_list = []
            for i in self.chromosome_list:
                file_prefix_i = re.sub(CHR_WILDCARD, str(i), self.file_prefix)
                geno_mul_y_i, snp_meta_i = self._calc_geno_mul_y(file_prefix_i, df_y)
                geno_mul_y_list.append(geno_mul_y_i)
                snp_meta_list.append(snp_meta_i)
            geno_mul_y = np.concatenate(geno_mul_y_list, axis=0)
            snp_meta = pd.concat(snp_meta_list, axis=0)
        return geno_mul_y, snp_meta
    def _calc_geno_mul_y(self, file_prefix, df_y):
        '''
        df_y is a pandas DataFrame where indiv_id contains "FID_IID" and y contains the values.
        Mean center genotype X and do X.T @ y.
        In principle, we also want to center y (center phenotype as well), 
        but here since X has been column-wise centered, centering y or not has no effect. 
        '''
        geno_mat, snp_meta, indiv_meta = self._load_geno_mat(file_prefix + '.bed')
        geno_mat = mean_center_col(geno_mat)
        # reorder rows in df_y so that it matches indiv_meta from the genotype
        df_y_reorder = pd.DataFrame({'indiv_id': indiv_meta})
        df_y_reorder = pd.merge(df_y_reorder, df_y, on='indiv_id', how='left')
        y_vals = df_y_reorder.y.values
        nmiss = np.isnan(y_vals).sum()
        # fill nan with zeros and it won't affect the result
        # y_vals[~np.isnan(y_vals)] = mean_center_col(y_vals[~np.isnan(y_vals)])
        y_vals[np.isnan(y_vals)] = 0
        if df_y_reorder.shape[0] - nmiss != df_y.shape[0]:
            warnings.warn('There are {}/{} individuals that are not in genotype file.'.format(df_y_reorder.shape[0] - nmiss, df_y.shape[0]))
        return geno_mat.T @ y_vals, snp_meta
    def _calc_geno_cov_per_file(self, file_prefix, mode, nbatch=2, params=None):
        geno_mat, snp_meta, _ = self._load_geno_mat(file_prefix + '.bed')
        constructor = CovConstructor(geno_mat, nbatch=nbatch)
        cov_mat = constructor.compute_cov(mode, param=params)
        return cov_mat, snp_meta
    def calc_geno_cov(self, nbatch=2, mode='banded', params=None):
        if self.read_by_chromosome is False:
            covmat, snp_meta = self._calc_geno_cov_per_file(file_prefix=self.file_prefix, mode=mode, params=params, nbatch=nbatch)
        else:
            covmat_list = []
            snp_meta_list = []
            for i in self.chromosome_list:
                file_prefix_i = re.sub(CHR_WILDCARD, str(i), self.file_prefix)
                covmat_i, snp_meta_i = self._calc_geno_cov_per_file(file_prefix=file_prefix_i, mode=mode, params=params)
                covmat_list.append(covmat_i)
                snp_meta_list.append(snp_meta_i)
            covmat = scipy.sparse.block_diag(covmat_list)
            snp_meta = pd.concat(snp_meta_list, axis=0)
        return covmat, snp_meta
    def _load_geno_mat(self, bedfile):
        geno_mat, indiv_list, _, snp_meta = load_genotype_from_bedfile(
            bedfile, snplist_to_exclude=self.snp_to_exclude, 
            missing_rate_cutoff=self.missing_rate_cutoff, return_snp=True,
            standardize=False
        )  
        return geno_mat, snp_meta, indiv_list