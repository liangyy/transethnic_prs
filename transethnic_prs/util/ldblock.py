import re
import pandas as pd

def load_ldetect(fn):
    df_ldblk = pd.read_csv(fn, sep='\s+')
    df_ldblk.chr = [ re.sub('chr', '', i) for i in df_ldblk.chr ]
    df_ldblk.rename(columns={'chr': 'chrom'}, inplace=True)
    return df_ldblk
