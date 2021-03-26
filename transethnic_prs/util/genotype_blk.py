import numpy as np
from tqdm import tqdm

def snplist_by_blk(df_ldblk, snp_meta, disable_progress_bar=True):
    out = []
    for i in tqdm(range(df_ldblk.shape[0]), disable=disable_progress_bar):
        chrm, s, e = df_ldblk.iloc[i, :]
        snp_meta_i = snp_meta[ 
            (snp_meta.chrom == chrm) & 
            ((snp_meta.pos < e) & (snp_meta.pos >= s)) 
        ]
        if snp_meta_i.shape[0] == 0:
            continue
        out.append(snp_meta_i.reset_index(drop=True))
    return out   

def _geno_by_blk_w_func(df_ldblk, geno_mat, snp_meta, func, disable_progress_bar=True):
    out = []
    geno_blk = []
    for i in tqdm(range(df_ldblk.shape[0]), disable=disable_progress_bar):
        chrm, s, e = df_ldblk.iloc[i, :]
        snp_meta_i = snp_meta[ 
            (snp_meta.chrom == chrm) & 
            ((snp_meta.pos < e) & (snp_meta.pos >= s)) 
        ]
        if snp_meta_i.shape[0] == 0:
            continue
        geno_sub = geno_mat[:, snp_meta_i.index.tolist()]
        out.append(func(geno_sub.T))
    return out   

def do_nothing(x):
    return x

def cov_w_stable_out_dim(x):
    tmp = np.cov(x)
    if tmp.shape == ():
        tmp = np.array([[tmp[0]]])
    return tmp

def geno_to_cov_by_blk(df_ldblk, geno_mat, snp_meta, disable_progress_bar=True):
    return _geno_by_blk_w_func(df_ldblk, geno_mat, snp_meta, cov_w_stable_out_dim, disable_progress_bar=disable_progress_bar)     

def geno_by_blk(df_ldblk, geno_mat, snp_meta, disable_progress_bar=True):
    return _geno_by_blk_w_func(df_ldblk, geno_mat, snp_meta, np.transpose, disable_progress_bar=disable_progress_bar)   
     