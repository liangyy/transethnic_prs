import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin

from transethnic_prs.util.misc import merge_two_lists

def load_genotype_from_bedfile(
    bedfile, 
    snplist_to_exclude=None, chromosome=None, 
    missing_rate_cutoff=0.5, return_snp=False, standardize=False
):
    
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
        pos = G.variant.pos.to_series().to_numpy()
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
            pos = pos[~np.isin(snpid, snplist_to_exclude)]
            chrom = chrom[~np.isin(snpid, snplist_to_exclude)]
        
        snpid = snpid[~np.isin(snpid, snplist_to_exclude)]
   
    # filter out genotypes with high missing rate
    missing_rate = np.isnan(geno).mean(axis=0)
    geno = geno[:, missing_rate < missing_rate_cutoff]
    if return_snp is True:
        snpid = snpid[missing_rate < missing_rate_cutoff]
        a0 = a0[missing_rate < missing_rate_cutoff]
        a1 = a1[missing_rate < missing_rate_cutoff]
        pos = pos[missing_rate < missing_rate_cutoff]
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
        pos = pos[to_keep]
        chrom = chrom[to_keep]
        
    maf = maf[to_keep]
    var_geno = var_geno[to_keep]
    if standardize is True:
        geno = (geno - 2 * maf) / np.sqrt(var_geno)
    
    if return_snp is True:
        df_snp = pd.DataFrame({
            'snpid': snpid.tolist(), 
            'a0': a0.tolist(), 
            'a1': a1.tolist(), 
            'chrom': chrom.tolist(),
            'pos': pos.tolist()
        })
        return geno, indiv_list, np.sqrt(var_geno), df_snp
    else:
        return geno, indiv_list, np.sqrt(var_geno)
