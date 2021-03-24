import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

from rpy2.robjects.conversion import localconverter
lassosum = importr('lassosum')
data_table = importr('data.table')
base = importr('base')
utils = importr('utils')

import numpy as np
import pandas as pd

from transethnic_prs.util.misc import scale_array_list, intersect_two_lists, get_index_of_l2_from_l1
from transethnic_prs.util.genotype_blk import geno_by_blk, geno_to_cov_by_blk, snplist_by_blk
from transethnic_prs.util.ldblock import load_ldetect, cmp_two_rng
from transethnic_prs.util.genotype_io import load_genotype_from_bedfile, get_complement

from transethnic_prs.model1.Model1Blk import Model1Blk

def load_data(pop1_bfile, pop2_bfile, ldblock_pop1, ldblock_pop2, first_nsnp=None):
    
    # load ld block
    ldblk1, ldblk_rng1 = load_ldetect(ldblock_pop1)
    ldblk2, ldblk_rng2 = load_ldetect(ldblock_pop2)
    if not cmp_two_rng(ldblk_rng1, ldblk_rng2):
        raise ValueError('LD block files do not have the same range for LD blocks.')
        
    # load genotype BED files
    geno1, _, _, snp1 = load_genotype_from_bedfile(pop1_bfile, return_snp=True)
    geno2, indiv2, _, snp2 = load_genotype_from_bedfile(pop2_bfile, return_snp=True)
    snp1['snpid'] = [ f'{ch}_{s}' for ch, s in zip(snp1.chrom, snp1.pos) ]
    snp2['snpid'] = [ f'{ch}_{s}' for ch, s in zip(snp2.chrom, snp2.pos) ]
    
    # get common snps
    snp1['is_amb'] = [ a1 == get_complement(a2) for a1, a2 in zip(snp1.a1, snp1.a0) ] 
    snp1_new = snp1[~snp1.is_amb].reset_index(drop=True)
    common_snp = intersect_two_lists(snp1_new.snpid, snp2.snpid)
    if first_nsnp is not None and first_nsnp < len(common_snp):
        common_snp = common_snp[:first_nsnp]
        # get sorted common snp
        common_snp = list(pd.DataFrame({
            'snpid': common_snp, 
            'chr': [ int(x.split('_')[0]) for x in common_snp],
            'pos': [ int(x.split('_')[1]) for x in common_snp]
        }).sort_values(by=['chr', 'pos']).snpid)
    idx1 = get_index_of_l2_from_l1(snp1.snpid, common_snp) 
    idx2 = get_index_of_l2_from_l1(snp2.snpid, common_snp)  
    
    # subset genotype to common snps
    geno1 = geno1[:, idx1]
    geno2 = geno2[:, idx2]
    snp1 = snp1.iloc[idx1, :].reset_index(drop=True)
    snp2 = snp2.iloc[idx2, :].reset_index(drop=True)
    
    # genotype/genotype cov by block
    geno_blk1 = geno_by_blk(ldblk1, geno1, snp1)
    geno_blk2 = geno_by_blk(ldblk2, geno2, snp2)
    geno_cov_blk1 = geno_to_cov_by_blk(ldblk1, geno1, snp1)
    geno_cov_blk2 = geno_to_cov_by_blk(ldblk2, geno2, snp2)
    geno_blk2_in_1 = geno_by_blk(ldblk1, geno2, snp2)
    snp_blk1 = snplist_by_blk(ldblk1, snp1)
    snp_blk2 = snplist_by_blk(ldblk2, snp2)
    snp_blk2_in_1 = snplist_by_blk(ldblk1, snp2)
    
    geno_blk_tuple = (geno_blk1, geno_blk2, geno_blk2_in_1)
    geno_cov_tuple = (geno_cov_blk1, geno_cov_blk2)
    snp_tuple = (snp_blk1, snp_blk2, snp_blk2_in_1)
    
    return  geno_blk_tuple, geno_cov_tuple, snp_tuple, indiv2
    
    
def model1_wrapper(pop1_xcovlist, pop1_gwas_list, pop1_N,
                   pop2_xlist, pop2_y, model1_kwargs, alpha=1, offset=[0, 0.1, 0.2, 0.3], method='full'):
    # model1
    alist = scale_array_list(pop1_xcovlist, pop1_N - 1)
    blist = [ bhat * covx.diagonal() * (pop1_N - 1) for bhat, covx in zip(pop1_gwas_list, pop1_xcovlist) ]
    mod1 = Model1Blk(
        Alist=alist,
        blist=blist,
        Xlist=pop2_xlist,
        y=pop2_y
    )
    beta_out = []
    lambda_out = []
    niter_out = []
    tol_out = []
    for o_  in offset:
        print('offset =', o_)
        if method == 'full':
            beta_mat, lambda_seq, niter, tol = mod1.solve_path(alpha=alpha, offset=o_ * (pop1_N - 1), **model1_kwargs)
        elif method == 'by_block':
            beta_mat, lambda_seq, niter, tol = mod1.solve_path_by_blk(alpha=alpha, offset=o_ * (pop1_N - 1), **model1_kwargs)
        beta_out.append(beta_mat[:, :, np.newaxis])
        lambda_out.append(lambda_seq[:, np.newaxis])
        niter_out.append(niter)
        tol_out.append(tol)
    return np.concatenate(beta_out, axis=2), np.concatenate(lambda_out, axis=1), niter_out, tol_out

def lassosum_wrapper(gwas_df, gwas_N, ref_bfile, ldblock, s_seq=[0.2, 0.5, 0.9, 1], lambda_max=None, lassosum_args={}):
    '''
    gwas_df = pd.DataFrame({
        'pval': pval,
        'bhat': bhat,
        'chrom': chrom,
        'pos': pos,
        'a1': 'a1',
        'a2': a2
    })
    ldblock: example 'EUR.hg19'
    '''
    with localconverter(ro.default_converter + pandas2ri.converter):
        gwas_rdf = ro.conversion.py2rpy(gwas_df)
    cor = lassosum.p2cor(
        p=rpy2_extract_col_from_df(gwas_rdf, 'pval'), 
        n=gwas_N, 
        sign=rpy2_extract_col_from_df(gwas_rdf, 'bhat')
    )
    
    if lambda_max is not None:
        lassosum_args['lambda'] = ro.FloatVector(np.exp(np.linspace(np.log(lambda_max / 100), np.log(lambda_max), num=20)))

    out = lassosum.lassosum_pipeline(
        cor=cor, 
        chr=rpy2_extract_col_from_df(gwas_rdf, 'chrom'), 
        pos=rpy2_extract_col_from_df(gwas_rdf, 'pos'), 
        A1=rpy2_extract_col_from_df(gwas_rdf, 'a1'), 
        A2=rpy2_extract_col_from_df(gwas_rdf, 'a2'), # A2 is not required but advised
        ref_bfile=ref_bfile,
        test_bfile=ref_bfile,
        LDblocks=ldblock,
        s=ro.FloatVector(s_seq),
        destandardize=True,
        **lassosum_args
    )
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        snp_df = ro.conversion.rpy2py(out.rx2('sumstats'))
        
    gwas_df['snpid'] = [ f'{ch}_{s}' for ch, s in zip(gwas_df.chrom, gwas_df.pos) ]
    snp_df['snpid'] = [ f'{ch}_{s}' for ch, s in zip(snp_df.chr, snp_df.pos) ]
    idx = get_index_of_l2_from_l1(gwas_df.snpid, snp_df.snpid) 
    
    beta_mat = []
    for i in s_seq:
        tmp = np.asarray(out.rx2('beta').rx2(str(i)))[:, ::-1]
        beta_ = np.zeros((gwas_df.shape[0], tmp.shape[1]))
        beta_[idx, :] = tmp
        beta_mat.append(beta_[:, :, np.newaxis])
    beta_mat = np.concatenate(beta_mat, axis=2)
    
    return beta_mat

def rpy2_extract_col_from_df(df, col):
    cols = np.array(base.colnames(df))
    idx = int(np.where(cols == col)[0][0])
    return df.rx2(idx + 1)