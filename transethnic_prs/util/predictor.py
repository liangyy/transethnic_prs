from collections import OrderedDict

import pandas as pd

import transethnic_prs.util.genotype_io as genoio
from transethnic_prs.util.misc import intersect_two_lists

class Predictor:
    def __init__(self, df_snp):
        self._init_beta(df_snp)
    def _init_beta(self, df_beta):
        tmp = df_beta.copy()
        tmp['idx'] = [ i for i in range(tmp.shape[0]) ]
        tmp.sort_values(
            by=['chrom', 'pos', 'a1', 'a2'], 
            inplace=True, ignore_index=True
        )
        snps = genoio.snpinfo_to_snpid(
            tmp.chrom, tmp.pos, tmp.a1, tmp.a2
        )
        tmp = tmp.iloc[ snps.idx, : ].reset_index(drop=True)
        tmp = pd.DataFrame({'snpid': snps.snpid, 'beta_idx': tmp.idx, 'direction': snps.direction, 'chrom': tmp.chrom})
        df_beta_dict = OrderedDict()
        for cc in tmp.chrom.unique():
             df_beta_dict[cc] = tmp[ tmp.chrom == cc ].drop(columns='chrom').reset_index(drop=True)
        self.df_beta_dict = df_beta_dict
    def _get_common_snps(self, loader):
        loader_snps = loader.get_snplist()
        snps_dict = OrderedDict()
        for cc in self.df_beta_dict.keys():
            kk = self.df_beta_dict[cc]
            snps = intersect_two_lists(kk.snpid, loader_snps.snpid)
            snps_dict[cc] = list(kk[ kk.snpid.isin(snps) ].snpid)
        return snps_dict
    def predict(self, beta_mat, geno_loader):
        snps_dict = self._get_common_snps(geno_loader)
        geno = geno_loader.load(snps)
        return self._predict(geno, beta_mat, snps_dict)
    def _predict(self, geno, beta_mat, snps_dict):
        '''
        snps in snps_dict has the same order as df_beta in df_beta_dict
        '''
        out = None
        nsnp = 0
        for cc in self.df_beta_dict.keys():
            kk = self.df_beta_dict[cc]
            beta_idx_sub = list(kk[ kk.snpid.isin(snps) ].beta_idx)
            if len(beta_idx_sub) == 0:
                continue
            nsnp += len(beta_idx_sub)
            beta_mat_sub = beta_mat[ beta_idx_sub, : ]
            if out is None:
                out = geno @ beta_mat_sub
            else:
                out += geno @ beta_mat_sub
        return out, nsnp
    
        
