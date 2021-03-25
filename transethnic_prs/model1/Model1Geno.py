'''
Here we solve the same problem as the one in Model1Blk.
ONLY solve_by_blk is implemented.
TODO: implement solve even though it cannot use multithreading.
But instead of requiring Alist and Xlist being np.dnarray in memory, 
it loads these arrays on the fly from PLINK BED file. 
CAUTION: We work with column mean centered genotype and y.
'''

from multiprocessing import get_context
import os
# Limit ourselves to single-threaded jax/xla operations to avoid thrashing. See
# https://github.com/google/jax/issues/743.
os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

import pandas as pd
import numpy as np

import transethnic_prs.util.math_jax as mj 
from transethnic_prs.util.misc import init_nested_list
import transethnic_prs.util.genotype_io as genoio
from transethnic_prs.model1.Model1Helper import *
from transethnic_prs.model1.Model1GenoHelper import *



from transethnic_prs.util.misc import check_np_darray, intersect_two_lists

    

class Model1Geno:
    '''
    snplist = [ snplist_1, ..., snplist_k ]
    blist = [ b1, ..., bk ]
    y = y
    pop1_bed = PLINK BED for A
    pop2_bed = PLINK BED for X
    snplist_i = pd.DataFrame({'chr', 'pos', 'a1', 'a2'})
    
    Internally, snp as saved as chrm_pos_A1_A2 where A1_A2 follows rules specified in transethnic_prs.util.genotyp_io.snpinfo_to_snpid
    '''
    def __init__(self, snplist, bhatlist, gwas_n_factor, df_y, pop1_bed, pop2_bed, nthreads=None):
        '''
        df_y: pd.DataFrame({
            'indiv': IID,
            'y': y
        })
        self.snplist
        self.blist  # gwas bhat
        self.y
        self.pop1_loader
        self.pop2_loader
        '''
        self.nthreads = 1 if nthreads is None or not isinstance(nthreads, int) else nthreads
        self._set_y_and_pop2_loader(pop2_bed, df_y)
        self._set_pop1_loader(pop1_bed)
        self.gwas_n_factor = gwas_n_factor
        self._set_snp_and_b(snplist, bhatlist)
        self._update_to_common_snplist()
        self.n_blk = len(self.snplist)
        self._set_varx1()
    def _set_y_and_pop2_loader(self, pop2_bed, df_y):
        loader = genoio.PlinkBedIO(pop2_bed)
        # load individual in bed
        bed_indiv = loader.get_indiv()
        df_bed = pd.DataFrame({
            'indiv': bed_indiv,
            'idx': [ i for i in range(len(bed_indiv)) ]
        })
        # intersect with individual in df_y
        df_bed_in_both = df_bed[ df_bed.indiv.isin(df_y.indiv) ].reset_index(drop=True)
        del df_bed
        # set loader to only load these individuals
        loader.set_indiv_idx(list(df_bed_in_both.idx))
        # intersect df_y to only these individuals IN ORDER!
        df_y = pd.merge(
            df_bed_in_both, df_y, on='indiv', how='left'
        )
        self.y = mj.mean_center_col_1d_jax(df_y.y.values)
        self.pop2_loader = loader
    def _set_pop1_loader(self, pop1_bed):
        self.pop1_loader = genoio.PlinkBedIO(pop1_bed)
    def _set_snp_and_b(self, snplist, bhatlist):
        if len(snplist) != len(bhatlist):
            raise ValueError('snplist and blist have different length.')
        # ns_list = []
        for i in range(len(snplist)):
            ns = snplist[i].shape[0]
            nb = check_np_darray(bhatlist[i], dim=1)
            if ns != nb:
                raise ValueError(f'The {i}th element in snplist and blist has un-matched shape {ns} != {nb}.')
            # ns_list.append(ns)
        self.snplist = snplist
        self.blist = bhatlist  
        # return ns_list
    def _update_to_common_snplist(self):
        '''
        Make sure snplist is ordered by chr and pos
        '''
        snp1 = self.pop1_loader.get_snplist()
        snp2 = self.pop2_loader.get_snplist()
        common_snp = intersect_two_lists(snp1.snpid, snp2.snpid)
        snplist_new = []
        blist_new = []
        for snps, bs in zip(self.snplist, self.blist):
            if len(set(snps.chr)) != 1:
                raise ValueError('SNP per block should on the same chromosome.')
            snp3 = genoio.snpinfo_to_snpid(snps.chr, snps.pos, snps.a1, snps.a2, return_complete=True)
            snp3 = snp3[ snp3.snpid.isin(common_snp) ].reset_index(drop=True)
            snp3 = snp3.sort_values(by=['chr', 'pos', 'a1', 'a2']).reset_index(drop=True)
            bs = bs[ snp3.idx ]
            snplist_new.append(list(snp3.snpid))
            blist_new.append(bs * snp3.direction.values)
        self.snplist = snplist_new
        self.blist = blist_new 
    def _set_varx1(self):
        args_by_worker = self._varx1_args()
        with get_context("spawn").Pool(self.nthreads) as pool:
            self.varx1 = pool.map(
                calc_varx_, args_by_worker
            ) 
    def kkt_beta_zero_multi_threads(self, alpha, nthreads=None):
        args_by_worker = self._kkt_args(alpha)
        nthreads = self.nthreads if nthreads is None else nthreads
        with get_context("spawn").Pool(nthreads) as pool:
            res = pool.map(
                kkt_beta_zero_per_blk_, args_by_worker
            )
        res = np.array(res)
        return list(res.max(axis=0))
    
    def solve_path_by_blk(self, alpha=0.5, offset=0, tol=1e-5, maxiter=1000, nlambda=100, ratio_lambda=100, nthreads=None):
        '''
        Same info as solve_path.
        But here we solve each block one at a time and combine at the end.
        
        Alpha and offset could be a list of numbers.
        Since IO is expensive, we will solve all combination within one IO pass. 
        '''
        
        if not isinstance(alpha, list):
            alpha = [ alpha ]
        if not isinstance(offset, list):
            offset = [ offset ]
        
        # set nthreads
        nthreads = self.nthreads if nthreads is None else nthreads
        
        # check input parameters
        for a_ in alpha:
            solve_path_param_sanity_check(a_, nlambda, ratio_lambda)
        
        lambda_max = self.kkt_beta_zero_multi_threads(alpha, nthreads=nthreads)
        lambda_seq = get_lambda_seq(lambda_max, nlambda, ratio_lambda)
        # add the first solution (corresponds to lambda = lambda_max)
        
        args_by_worker = self._solve_path_by_snplist(
            alpha=alpha,
            lambda_seq=lambda_seq, 
            offset=offset, 
            tol=tol, 
            maxiter=maxiter
        )
        with get_context("spawn").Pool(nthreads) as pool:
            res = pool.map(
                solve_path_by_snplist__, args_by_worker
            ) 
        
        beta_list = init_nested_list(len(alpha), len(offset))
        niter_list = init_nested_list(len(alpha), len(offset))
        tol_list = init_nested_list(len(alpha), len(offset))

        for i in range(len(alpha)):
            for j in range(len(offset)):
                for b, n, t in res:
                    beta_list[i][j].append(b[i][j])
                    niter_list[i][j].append(n[i][j])
                    tol_list[i][j].append(t[i][j])
                beta_list[i][j] = np.concatenate(beta_list[i][j], axis=0)
                
        return beta_list, lambda_seq, niter_list, tol_list
    
    def _varx1_args(self):
        o = []
        for i in range(self.n_blk):
            o.append((self.pop1_loader, self.snplist[i]))
        return o    
    def _kkt_args(self, alpha):
        o = []
        for i in range(self.n_blk):
            o.append((self.pop2_loader, self.blist[i], self.gwas_n_factor, self.varx1[i], self.snplist[i], self.y, alpha))
        return o
    def _solve_path_by_snplist(self, alpha, lambda_seq, offset, tol, maxiter):
        o = []
        for i in range(self.n_blk):
            o.append(
                {
                    'snplist': self.snplist[i], 
                    'lambda_seq_list': lambda_seq, 
                    'alpha_list': alpha, 
                    'offset_list': offset, 
                    'tol': tol, 
                    'maxiter': maxiter,
                    'data_args': {
                        'loader1': self.pop1_loader, 
                        'loader2': self.pop2_loader, 
                        'gwas_n_factor': self.gwas_n_factor, 
                        'gwas_bhat': self.blist[i],
                        'y': self.y
                    }
                }
            )
        return o   