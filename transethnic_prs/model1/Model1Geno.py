'''
Here we solve the same problem as the one in Model1Blk.
ONLY solve_by_blk is implemented.
TODO: implement solve even though it cannot use multithreading.
But instead of requiring Alist and Xlist being np.dnarray in memory, 
it loads these arrays on the fly from PLINK BED file. 
CAUTION: We work with column mean centered genotype and standardized y (std = 1).
'''

from collections import OrderedDict
from copy import deepcopy

from multiprocessing import Pool

import pandas as pd
import numpy as np

import transethnic_prs.util.math_numba as mn 
from transethnic_prs.util.misc import init_nested_list
import transethnic_prs.util.genotype_io as genoio
from transethnic_prs.model1.Model1Helper import *
from transethnic_prs.model1.Model1GenoHelper import *



from transethnic_prs.util.misc import check_np_darray, intersect_two_lists

class Model1Geno:
    '''
    snplist = [ snplist_1, ..., snplist_k ]
    df_gwas = pd.DataFrame({
        'bhat': [ b1, ..., bk ],
        'se': [ se1, ..., sek ],
        'chr': [ ... ],
        'pos': [ ... ],
        'a1': [ ... ],
        'a2': [ ... ]
    })
    y = standardized(y)
    pop1_bed = PLINK BED for A
    pop2_bed = PLINK BED for X
    snplist_i = pd.DataFrame({'chr', 'pos', 'a1', 'a2'})
    
    Internally, snp as saved as chrm_pos_A1_A2 where A1_A2 follows rules specified in transethnic_prs.util.genotyp_io.snpinfo_to_snpid
    '''
    def __init__(self, snplist, df_gwas, gwas_sample_size, df_y, pop1_bed, pop2_bed, nthreads=1, no_gwas=False):
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
        self.nthreads = self._check_n_return_nthreads(nthreads)
        self.no_gwas = no_gwas
        self._set_y_and_pop2_loader(pop2_bed, df_y)
        self._set_pop1_loader(pop1_bed)
        self.gwas_sample_size = gwas_sample_size
        self._set_snp_and_z(snplist, df_gwas)
        self._update_to_common_snplist()
        self.n_blk = len(self.snplist)
        self._set_varx1()
        self._set_bhat()
    @staticmethod
    def _check_n_return_nthreads(nthreads):
        if isinstance(nthreads, int):
            return nthreads
        else:
            raise TypeError('nthreads can only be integer')
    def _return_threads_for_now(self, nthreads):
        if nthreads is not None:
            nthreads = self._check_n_return_nthreads(nthreads)
        else:
            nthreads = self.nthreads
        return nthreads
    def _set_y_and_pop2_loader(self, pop2_bed, df_y):
        if isinstance(pop2_bed, str):
            loader = genoio.PlinkBedIO(pop2_bed)
        elif isinstance(pop2_bed, genoio.PlinkBedIO):
            loader = pop2_bed
        else:
            raise TypeError('Wrong pop2_bed')
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
        self.y = mn.standardize_1d_numba(df_y.y.values)
        self.pop2_loader = loader
    def _set_pop1_loader(self, pop1_bed):
        if isinstance(pop1_bed, str):
            self.pop1_loader = genoio.PlinkBedIO(pop1_bed)
        elif isinstance(pop1_bed, genoio.PlinkBedIO):
            self.pop1_loader = pop1_bed
        else:
            raise TypeError('Wrong pop1_bed')
    @staticmethod
    def _assign_snpid(df):
        df_ = genoio.snpinfo_to_snpid(df.chr, df.pos, df.a1, df.a2)
        df = df.iloc[df_.idx, :].reset_index(drop=True)
        df['snpid'] = df_.snpid
        df['direction'] = df_.direction
        return df
    def _set_snp_and_z(self, snplist, df_gwas):
        # z = bhat / se 
        # bhat_std = z / sqrt(N) / sqrt(Vx) 
        # evaluate bhat_std later (after computing Vx)
        df_gwas['z'] = df_gwas.bhat.values / df_gwas.se 
        df_gwas = self._assign_snpid(df_gwas)
        zlist = []
        snplist_ = []
        for i in range(len(snplist)):
            ns = snplist[i].shape[0]
            snp_ = self._assign_snpid(snplist[i])
            snp_ = pd.merge(snp_[['snpid', 'direction']], df_gwas, on='snpid')
            if snp_.shape[0] == 0:
                continue
            zlist.append(snp_.z.values * snp_.direction_x.values * snp_.direction_y.values)
            snplist_.append(snp_[['chr', 'pos', 'a1', 'a2']].copy())
        self.snplist = snplist_
        self.blist = zlist  
    def _update_to_common_snplist(self):
        '''
        Make sure snplist is ordered by chr and pos
        '''
        snp1 = self.pop1_loader.get_snplist()
        snp2 = self.pop2_loader.get_snplist()
        common_snp = intersect_two_lists(snp1.snpid, snp2.snpid)
        snplist_new = []
        blist_new = []
        for snps, zs in zip(self.snplist, self.blist):
            if len(set(snps.chr)) != 1:
                raise ValueError('SNP per block should on the same chromosome.')
            snp3 = genoio.snpinfo_to_snpid(snps.chr, snps.pos, snps.a1, snps.a2, return_complete=True)
            snp3 = snp3[ snp3.snpid.isin(common_snp) ].reset_index(drop=True)
            snp3 = snp3.sort_values(by=['chr', 'pos', 'a1', 'a2']).reset_index(drop=True)
            zs = zs[ snp3.idx ]
            snplist_new.append(list(snp3.snpid))
            blist_new.append(zs * snp3.direction.values)
        self.snplist = snplist_new
        self.blist = blist_new 
    def _set_varx1(self):
        args_by_worker = self._varx1_args()
        if self.nthreads == 1:
            self.varx1 = []
            for args in args_by_worker:
                self.varx1.append(calc_varx_(args))
        else:
            with Pool(self.nthreads) as pool:
                self.varx1 = pool.map(
                    calc_varx_, args_by_worker
                ) 
    def _set_bhat(self):
        for i in range(len(self.blist)):
            if self.no_gwas is False:
                self.blist[i] = self.blist[i] / np.sqrt(self.gwas_sample_size - 1) / np.sqrt(self.varx1[i])    
            else:
                self.blist[i] = np.zeros(self.blist[i].shape)
    def kkt_beta_zero_multi_threads(self, alpha, w_dict, nthreads=None):
        args_by_worker = self._kkt_args(alpha, w_dict)
        nthreads = self._return_threads_for_now(nthreads)
        if nthreads == 1:
            res = []
            for args in args_by_worker:
                res.append(kkt_beta_zero_per_blk_(args))
        else:
            with Pool(nthreads) as pool:
                res = pool.map(
                    kkt_beta_zero_per_blk_, args_by_worker
                )
        
        
        # res = Parallel(n_jobs=self.nthreads, backend='threading')(
        #     delayed(kkt_beta_zero_per_blk_)(argi) for argi in args_by_worker
        # )
        out_dict = OrderedDict()
        for k in w_dict.keys():
            tmp = np.array([ res_[k] for res_ in res ])
            out_dict[k] = list(tmp.max(axis=0))
        return out_dict
    
    def solve_path_by_blk(self, alpha=0.5, offset=0, w=1., tol=1e-5, maxiter=1000, nlambda=100, ratio_lambda=100, nthreads=None, mode=None, lambda_seq=None, message=0):
        '''
        Same info as solve_path.
        But here we solve each block one at a time and combine at the end.
        
        Alpha and offset could be a list of numbers.
        Since IO is expensive, we will solve all combination within one IO pass. 
        
        Objective:
        
            (1 - offset) x' A x - 2 b' x + offset x' diag(A) x + w || y2 - X2 x ||_2^2 + penalty(x)
        
        Equiv. to:
        
            x' A x - 2 b' x + offset_ x' diag(A) x + w_ || y2 - X2 x ||_2^2 + penalty(x)
        
        with:
            
            offset_ = offset / (1 - offset)
            w_ = w / (1 - offset)
        '''
        
        if not isinstance(alpha, list):
            alpha = [ alpha ]
        if not isinstance(offset, list):
            offset = [ offset ]
            for o in offset:
                if o >= 1 or o < 0:
                    raise ValueError('Values in offset should be in [0, 1)')
        if not isinstance(w, list):
            w = [ w ]
        
        # rescale params
        offset_x_w = init_nested_list(len(offset), len(w))  # dim0: offset, dim1: w
        w_dict = OrderedDict()
        for io, o_ in enumerate(offset):
            for iw, w_ in enumerate(w):
                w_new = w_ / (1 - o_)
                offset_x_w[io][iw] = [ o_ / (1 - o_), w_new ]
                if w_new not in w_dict:
                    w_dict[w_new] = 1  # place holder for lambda_seq
        
        # set nthreads
        nthreads = self.nthreads if nthreads is None else nthreads
        
        # check input parameters
        for a_ in alpha:
            solve_path_param_sanity_check(a_, nlambda, ratio_lambda)
        
        if lambda_seq is None:
            lambda_max = self.kkt_beta_zero_multi_threads(alpha, w_dict=w_dict, nthreads=nthreads)
            lambda_seq = get_lambda_seq(lambda_max, nlambda, ratio_lambda)
        elif isinstance(lambda_seq, list):
            lambda_seq_ = OrderedDict()
            for w in w_dict.keys():
                lambda_seq_[w] = deepcopy(lambda_seq)
            lambda_seq = lambda_seq_
        elif isinstance(lambda_seq, OrderedDict):
            for w in w_dict.keys():
                if w not in lambda_seq:
                    raise ValueError(f'Invalid lambda_seq since it misses w = {w}')
        # add the first solution (corresponds to lambda = lambda_max)
        
        args_by_worker = self._solve_path_by_snplist(
            alpha=alpha,
            lambda_seq=lambda_seq, 
            offset_x_w=offset_x_w, 
            tol=tol, 
            maxiter=maxiter,
            mode=mode, 
            message=message
        )
        
        nthreads = self._return_threads_for_now(nthreads)
        if nthreads == 1:
            res = []
            for args in args_by_worker:
                res.append(solve_path_by_snplist__(args))
        else:
            with Pool(nthreads) as pool:
                res = pool.map(
                    solve_path_by_snplist__, args_by_worker
                ) 
        
        # nested list: offset, w, alpha
        beta_list = init_nested_list(
            len(offset_x_w), len(offset_x_w[0]), len(alpha)
        )
        niter_list = init_nested_list(
            len(offset_x_w), len(offset_x_w[0]), len(alpha)
        )
        tol_list = init_nested_list(
            len(offset_x_w), len(offset_x_w[0]), len(alpha)
        )
        conv_list = init_nested_list(
            len(offset_x_w), len(offset_x_w[0]), len(alpha)
        )

        for oi in range(len(offset_x_w)):
            for wi in range(len(offset_x_w[0])):
                for ai in range(len(alpha)):
                    for b, n, t, conv in res:
                        beta_list[oi][wi][ai].append(b[oi][wi][ai])
                        niter_list[oi][wi][ai].append(n[oi][wi][ai])
                        tol_list[oi][wi][ai].append(t[oi][wi][ai])
                        conv_list[oi][wi][ai].append(conv[oi][wi][ai])
                    beta_list[oi][wi][ai] = np.concatenate(beta_list[oi][wi][ai], axis=0)
                
        return beta_list, lambda_seq, niter_list, tol_list, conv_list
    
    def _varx1_args(self):
        o = []
        for i in range(self.n_blk):
            o.append((self.pop1_loader, self.snplist[i]))
        return o    
    def _kkt_args(self, alpha, w_dict):
        o = []
        for i in range(self.n_blk):
            o.append((self.pop2_loader, self.blist[i], self.gwas_sample_size - 1, self.varx1[i], self.snplist[i], self.y, alpha, w_dict))
        return o
    def _solve_path_by_snplist(self, alpha, lambda_seq, offset_x_w, tol, maxiter, mode, message):
        o = []
        for i in range(self.n_blk):
            o.append(
                {
                    'blk_idx': i,
                    'snplist': self.snplist[i], 
                    'lambda_seq_dict': lambda_seq, 
                    'alpha_list': alpha, 
                    'offset_x_w_list': offset_x_w, 
                    'tol': tol, 
                    'maxiter': maxiter,
                    'mode': mode,
                    'message': message,
                    'data_args': {
                        'loader1': self.pop1_loader, 
                        'loader2': self.pop2_loader, 
                        'gwas_sample_size': self.gwas_sample_size, 
                        'gwas_bhat': self.blist[i],
                        'y': self.y
                    }
                }
            )
        return o   
    def get_snps(self):
        snps = []
        for i in self.snplist:
            snps += i
        return genoio.parse_snpid(snps)
