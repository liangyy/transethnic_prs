from collections import OrderedDict

import numpy as np
import pandas as pd
import string, re
from tqdm import tqdm

from pandas_plink import read_plink1_bin

from transethnic_prs.util.misc import merge_two_lists, list_is_equal

COMPLEMENT_BASE = {
    'A': 'T',
    'G': 'C',
    'T': 'A',
    'C': 'G'
}
CHR_TO_ORDER = { k: i for i, k in enumerate(string.ascii_uppercase) }
SNPID_SEP = ':'
CHRNUM_WILDCARD='{chr_num}'

class PlinkBedIO:
    def __init__(self, bedfile, chromosomes=None, show_progress_bar=False):
        '''
        chromosomes only effective when bedfile contains CHRNUM_WILDCARD
        '''
        self.bedfile_pattern = bedfile
        self._set_chromosomes(chromosomes)
        self._set_indiv()
        self._set_snplist(show_progress_bar)
    def get_bedfile(self, chrom=None):
        if CHRNUM_WILDCARD in self.bedfile_pattern:
            if chrom is None:
                chrom = self.chromosomes[0]
            return re.sub(CHRNUM_WILDCARD, str(chrom), self.bedfile_pattern)
        else:
            return self.bedfile_pattern
    def _set_chromosomes(self, chroms):
        if CHRNUM_WILDCARD in self.bedfile_pattern:
            if chroms is None:
                self.chromosomes = [ i for i in range(1, 23) ]
            else:
                self.chromosomes = chroms
        else:
            self.chromosomes = None
    def _set_indiv(self):
        indiv_list = load_indiv(self.get_bedfile())
        if CHRNUM_WILDCARD in self.bedfile_pattern:
            for cc in self.chromosomes[1:]:
                indiv_new = load_indiv(self.get_bedfile(cc))
                if list_is_equal(indiv_new, indiv_list):
                    continue
                else:
                    raise ValueError('When using wildcard, all chromosomes should have exactly the same individual list including the order.')
        self.indiv_all = np.array(indiv_list)
        self.indiv_active_idx = np.array([ i for i in range(len(indiv_list))])
    def _set_snplist(self, show_progress_bar=False):
        snplist = load_snplist(self.get_bedfile())
        if CHRNUM_WILDCARD in self.bedfile_pattern:
            snplist_dict = OrderedDict()
            snplist_dict[self.chromosomes[0]] = snplist
            for cc in tqdm(self.chromosomes[1:], disable=not show_progress_bar):
                snplist_dict[cc] = load_snplist(self.get_bedfile(cc))
            snplist = snplist_dict
        self.snp_all = snplist
    def get_snplist(self):
        if isinstance(self.snp_all, OrderedDict):
            snp = []
            for _, v in self.snp_all.items():
                snp.append(v)
            return pd.concat(snp, axis=0)
        else:
            return self.snp_all
    def get_indiv(self):
        return self.indiv_all[self.indiv_active_idx]
    def set_indiv_idx(self, indiv_idx):
        if min(indiv_idx) < 0 or max(indiv_idx) > self.indiv_all.shape[0]:
            raise ValueError('Elements in indiv_idx should within 0 ~ (ninvid - 1).')
        if len(set(indiv_idx)) != len(indiv_idx):
            raise ValueError('There are duplicated values in indiv_idx.')
        self.indiv_active_idx = indiv_idx
    def _get_snp_idx(self, snps):
        if isinstance(self.snp_all, OrderedDict):
            chrm = int(snps[0].split(SNPID_SEP)[0])
            snplist = self.snp_all[chrm]
            bedfile = self.get_bedfile(chrm)
        else:
            snplist = self.snp_all
            bedfile = self.get_bedfile()
        snps = snplist[ snplist.snpid.isin(snps) ].reset_index(drop=True)
        return snps, bedfile
    def load(self, snps, standardize=False, missing_rate_warn_cutoff=0.5, return_snplist=False):
        df_active_snp, bedfile = self._get_snp_idx(snps)
        G = read_plink1_bin(bedfile, verbose=False)
        geno = G.sel(
            sample=self.get_indiv(), 
            variant=[ 'variant' + str(i) for i in df_active_snp.idx ]
        ).values
        
        # flip
        # 2 * is_flip + geno * direction
        vec2 = np.ones((1, geno.shape[1])) * 2
        dire = df_active_snp.direction.values[np.newaxis, :]
        geno = vec2 * (dire == -1) + geno * dire
        
        # missing rate of snps
        missing_rate = np.isnan(geno).mean(axis=0)
        if missing_rate.max() > missing_rate_warn_cutoff:
            warnings.warn(f'Highest missing rate is greater than {missing_rate_warn_cutoff}.')
        # impute genotype missing value
        maf = np.nanmean(geno, axis=0) / 2
        miss_x, miss_y = np.where(np.isnan(geno))
        geno[(miss_x, miss_y)] = maf[miss_y] * 2
        if standardize is True:
            var_geno = 2 * maf * (1 - maf)
            geno = (geno - 2 * maf) / np.sqrt(var_geno)
        # TODO: is copy necessary? it is added to ensure C array
        if return_snplist is True:
            return geno.copy(), df_active_snp
        else:
            return geno.copy()

def parse_snpid(snp):
    chrm, pos, a1, a2 = [], [], [], []
    for i in snp:
        cc, pp, aa1, aa2 = i.split(SNPID_SEP)
        chrm.append(int(cc))
        pos.append(int(pp))
        a1.append(aa1)
        a2.append(aa2)
    return pd.DataFrame({'chrom': chrm, 'pos': pos, 'a1': a1, 'a2': a2})
        
def get_complement(str_):
    o = ''
    for s in str_:
        char_ = s.upper()
        if char_ not in COMPLEMENT_BASE:
            raise ValueError(f'Wrong s in str_: s = {s}.')
        o = COMPLEMENT_BASE[char_] + o
    return o

def get_order(str_):
    o = 0
    for s in str_:
        char_ = s.upper()
        if char_ in CHR_TO_ORDER:
            o = o * 10 + CHR_TO_ORDER[char_]
        else:
            raise ValueError(f'Wrong s in str_: s = {s}.')
    return o

def give_unique_pair(a1, a2):
    # normal order
    o1 = get_order(a1)
    # flip order 
    o2 = get_order(a2)
    # complement normal order
    ca1 = get_complement(a1)
    o3 = get_order(ca1)
    # complement flip order
    ca2 = get_complement(a2)
    o4 = get_order(ca2)
    if o1 <= min(o2, o3, o4):
        return a1 + SNPID_SEP + a2, 1
    elif o2 < min(o1, o3, o4):
        return a2 + SNPID_SEP + a1, -1
    elif o3 <= min(o1, o2, o4):
        return ca1 + SNPID_SEP + ca2, 1
    elif o4 < min(o1, o2, o3):
        return ca2 + SNPID_SEP + ca1, -1
    else:
        raise ValueError(f'''
            Something wrong: 
                o1 = {o1}, 
                o2 = {o2}, 
                o3 = {o3},
                o4 = {o4},
                a1 = {a1},
                a2 = {a2}
        ''')
    
def snpinfo_to_snpid(chrm, pos, a1, a2, return_complete=False, allow_ambi=False):
    '''
    Internally, we convert chrm, pos, a1, a2 as chrm_pos_A1_A2, 
    where A1 and A2 follow the rule that A1A2 < all other flip and/or complement combinations.
    Ambiguious SNPs are discarded!
    Return: pd.DataFrame({
        'idx': snp idx (only keep non-ambiguous snps) 
        'snpid': snpid,
        'direction': if flip ref and alt alleles, set to -1; otherwise, 1
    })
    '''
    snpids = []
    directions = []
    idxs = []
    idx = -1
    if return_complete is True:
        clist, plist, a1list, a2list = [], [], [], []
    for c, p, A1, A2 in zip(chrm, pos, a1, a2):
        idx += 1
        if allow_ambi is False and A1.upper() == get_complement(A2):
            continue
        allele_pair, direction = give_unique_pair(A1, A2)
        snpids.append(SNPID_SEP.join([ str(c), str(p), allele_pair ]))
        directions.append(direction)
        idxs.append(idx)
        if return_complete is True:
            clist.append(c)
            plist.append(p)
            a1list.append(A1)
            a2list.append(A2)
    if return_complete is False:
        return pd.DataFrame({'idx': idxs, 'snpid': snpids, 'direction': directions})
    else:
        return pd.DataFrame({'idx': idxs, 'snpid': snpids, 'direction': directions, 'chr': clist, 'pos': plist, 'a1': a1list, 'a2': a2list})
def load_snplist(bedfile, return_complete=False):
    G = read_plink1_bin(bedfile, verbose=False)
    a2 = list(G.variant.a0.to_series())
    a1 = list(G.variant.a1.to_series()) 
    pos = list(G.variant.pos.to_series())
    chrom = list(G.variant.chrom.to_series())
    return snpinfo_to_snpid(chrom, pos, a1, a2, return_complete=return_complete)

def load_indiv(bedfile):
    G = read_plink1_bin(bedfile, verbose=False)
    iid = G.iid.to_series().tolist()
    return iid

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
    
    # keep only genotypes with variance != 0
    var_geno = np.var(geno, axis=0)  # 2 * maf * (1 - maf)
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
        '''
        a1 is the effect allele
        '''
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
