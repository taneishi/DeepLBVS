import pandas as pd
import numpy as np
import gzip
import os
import sys

COMPLEVEL = 5

def profeat():
    if os.path.exists('pkl/profeat'):
        return
    df = pd.read_csv('data/profeat_out.txt.gz', sep='\t', header=None, index_col=0)
    df = df[df.index != 'deleted']
    df = df.astype(np.float32)

    name = pd.read_csv('data/protname_out.txt.gz', sep='\t', header=None)
    name = name[[0, 3]].rename(columns={0:'uid', 3:'pid'})

    df = pd.merge(df, name, left_index=True, right_on='pid', how='left')

    df.ix[df['uid'].isnull(), 'uid'] = df.ix[df['uid'].isnull(), 'pid']
    df = df[df['uid'].str.endswith('_HUMAN')]

    df = df.set_index('uid')
    del df['pid']
    del df.index.name
    df = df.drop_duplicates()
    df.columns = df.columns.map(lambda x: 'P%d' % x)
    df.to_pickle('pkl/profeat')

def dragon():
    if os.path.exists('pkl/dragon'):
        return
    df = pd.read_csv('data/dragon_out.txt.gz', sep='\t', header=None, index_col=0)
    df = df.astype(np.float32)
    del df.index.name
    df.columns = df.columns.map(lambda x: 'C%d' % x)
    df.to_pickle('pkl/dragon')

def dataset():
    np.random.seed(123)
    
    profeat = pd.read_pickle('pkl/profeat')
    dragon = pd.read_pickle('pkl/dragon')

    posi = pd.read_csv(gzip.open('tsv/posi_gpcr_full.pair.gz'), sep='\t', header=None)
    posi['label'] = 1

    for i in range(1, 6):
        nega = pd.read_csv(gzip.open('tsv/nega_gpcr_full_%d.out.gz' % i), sep='\t', header=None)
        nega['label'] = 0

        df = pd.concat([posi,nega])
        df = df.rename(columns={0:'cid', 1:'pid'})
        df = pd.merge(dragon, df, left_index=True, right_on='cid', how='inner')
        df = pd.merge(profeat, df, left_index=True, right_on='pid', how='inner')

        df.set_index(['cid','pid'], inplace=True)

        df = df.astype(np.float32)
        df.to_pickle('pkl/dataset%d.pkl' % i)

if __name__ == '__main__':
    profeat()
    dragon()
    dataset()
