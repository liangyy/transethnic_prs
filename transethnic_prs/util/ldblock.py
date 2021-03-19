import re
import pandas as pd

import warnings

def load_ldetect(fn):
    df_ldblk = pd.read_csv(fn, sep='\s+')
    df_ldblk.chr = [ re.sub('chr', '', i) for i in df_ldblk.chr ]
    df_ldblk.rename(columns={'chr': 'chrom'}, inplace=True)
    
    # sometimes there are 'None' in ldetect file
    n_none = ((df_ldblk.start == 'None') | (df_ldblk.stop == 'None')).sum()
    if n_none != 0:
        df_ldblk = _ugly_merge_ldetect(df_ldblk)      
    df_ldblk.start = df_ldblk.start.astype(int)
    df_ldblk.stop = df_ldblk.stop.astype(int)
    # convert base 0 bed format to base 1 start and end (not contained)
    df_ldblk.start = df_ldblk.start + 1
    df_ldblk.stop = df_ldblk.stop + 1
    return df_ldblk, _sanity_check_ldetect(df_ldblk)

def cmp_two_rng(rng1, rng2):
    k1, k2 = set(rng1.keys()), set(rng2.keys())
    if k1.issubset(k2) and k2.issubset(k1):
        pass
    else:
        warnings.warn('Different chromosomes.')
        return False
    for k in k1:
        s1, e1 = rng1[k]
        s2, e2 = rng2[k]
        if s1 != s2 or e1 != e2:
            warnings.warn(f'For chromosome {k}: s1 != s2 or e1 != e2.')
            return False
    return True

def _sanity_check_ldetect(df_ldblk):
    '''
    Check if df_ldblk is ordered by chr and position.
    And also check if the regions are exactly consecutive.
    Return a dictionary containing the range for each chromosome
    '''
    range_dict = {}
    curr_last = None
    pre_chr = None
    for i in range(df_ldblk.shape[0]):
        chr, s, e = df_ldblk.iloc[i, :]
        if chr not in range_dict:
            range_dict[chr] = [ s ]
            if pre_chr is not None:
                range_dict[pre_chr].append(curr_last)
            pre_chr = chr
            curr_last = e
        else:
            if s != curr_last:
                raise ValueError(f'Wrong format at row = {i}: s[0] != e[-1].')
            curr_last = e
    range_dict[pre_chr].append(curr_last)
    return range_dict
        
            

def _ugly_merge_ldetect(df_ldblk):
    df_new = []
    i = 0
    while i < df_ldblk.shape[0]:
        s, e = df_ldblk.iloc[i, 1:]
        if s != 'None' and e != 'None':
            df_new.append(df_ldblk.iloc[i, :])
            i += 1
            continue
        else:
            if e == 'None':
                tmp = df_ldblk.iloc[i, :]
                i += 1
                while True:
                    s2, e2 = df_ldblk.iloc[i, 1:]
                    if s2 != 'None':
                        raise ValueError('Wrong format: e[0] = None and s[+1] != None.')
                    else:
                        i += 1
                        if e2 == 'None':
                            continue
                        else:
                            e = e2
                            break
                tmp.stop = e
                df_new.append(tmp.to_frame())
                continue
            else:
                raise ValueError('Wrong format: s[0] = None and e[-1] != None.')
    return pd.concat(df_new, axis=1).T
