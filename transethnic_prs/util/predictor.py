import transethnic_prs.util.genotype_io as genoio
from transethnic_prs.util.misc import intersect_two_lists

class Predictor:
    def __init__(self, df_beta):
        self._init_beta(df_beta)
    def _init_beta(self, df_beta):
        snps = genoio.snpinfo_to_snpid(
            df_beta.chrom, df_beta.pos, df_beta.a1, df_beta.a2
        )
        df_beta.sort_values(
            by=['chrom', 'pos', 'a1', 'a2'], 
            inplace=True, ignore_index=True
        )
        df_beta = df_beta.iloc[ snps.idx, : ].reset_index(drop=True)
        df_beta.beta *= snps.direction.values
        self.df_beta = pd.DataFrame({'beta': df_beta.beta.values, 'snpid': snps.snpid})
    def _get_common_snps(self, loader):
        loader_snps = loader.get_snplist()
        snps = intersect_two_lists(self.df_beta.snpid, loader_snps)
        return list(self.df_beta[ self.df_beta.snpid.isin(snps) ].snpid)
    def predict(self, geno_loader):
        snps = self._get_common_snps(geno_loader)
        geno = geno_loader.load(snps)
        return self._predict(geno, snps)
    def _predict(self, geno, snps):
        '''
        snps has the same order as self.df_beta
        '''
        beta = self.df_beta[ self.df_beta.snpid.isin(snps) ].beta.values
        return geno @ beta
    
        