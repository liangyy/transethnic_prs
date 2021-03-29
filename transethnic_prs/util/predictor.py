import transethnic_prs.util.genotype_io as genoio
from transethnic_prs.util.misc import intersect_two_lists

class Predictor:
    def __init__(self, df_snp):
        self._init_beta(df_snp, beta_mat)
    def _init_beta(self, df_beta, beta_mat):
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
        self.df_beta = pd.DataFrame({'snpid': snps.snpid, 'beta_idx': tmp.idx, 'direction': snps.direction})
    def _get_common_snps(self, loader):
        loader_snps = loader.get_snplist()
        snps = intersect_two_lists(self.df_beta.snpid, loader_snps)
        return list(self.df_beta[ self.df_beta.snpid.isin(snps) ].snpid)
    def predict(self, beta_mat, geno_loader):
        snps = self._get_common_snps(geno_loader)
        geno = geno_loader.load(snps)
        return self._predict(geno, beta_mat, snps)
    def _predict(self, geno, snps):
        '''
        snps has the same order as self.df_beta
        '''
        beta_idx_sub = list(self.df_beta[ self.df_beta.snpid.isin(snps) ].beta_idx)
        beta_mat_sub = beta_mat[ beta_idx_sub, : ]
        return geno @ beta_mat_sub
    
        