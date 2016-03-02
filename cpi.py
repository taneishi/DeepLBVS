# coding:utf-8
import pandas as pd
import numpy as np
import gzip
import os

COMPLEVEL = 5

def profeat():
    dtype = dict([(i,np.float32) for i in range(1,1081)])
    df = pd.read_csv('profeat_out.txt',sep='\t',header=None,dtype=dtype)
    df = df[df[0] != 'deleted']
    df = df.set_index(0)
    name = pd.read_csv('protname_out.txt',sep='\t',header=None)
    name.columns = ['uid','','pname','pid']

    df = pd.merge(df, name[['uid','pid']], left_index=True, right_on='pid', how='left')
    cond = df['uid'].isnull()
    df.ix[cond, 'uid'] = df.ix[cond, 'pid']

    df = df[df['uid'].str.endswith('_HUMAN')]

    df = df.set_index('uid')
    del df.index.name
    del df['pid']
    df = df.drop_duplicates()
    df.to_hdf('cpi.hdf', 'profeat', format='fixed', complevel=COMPLEVEL, complib='zlib')

def dragon():
    dtype = dict([(i,np.float32) for i in range(1,895)])
    df = pd.read_csv('dragon_out.txt', sep='\t', header=None, dtype=dtype)
    df = df.set_index(0)

    del df.index.name
    df.to_hdf('cpi.hdf', 'dragon', format='fixed', complevel=COMPLEVEL, complib='zlib')

def dataset():
    posi = pd.read_csv(gzip.open('tsv/posi_gpcr_full.pair.gz'),sep='\t',header=None)
    posi['label'] = 1
    posi.to_hdf('cpi.hdf', 'posi')

    for i in range(1, 6):
        nega = pd.read_csv(gzip.open('tsv/nega_gpcr_full_%d.out.gz' % i),sep='\t',header=None)
        nega['label'] = 0
        nega.to_hdf('cpi.hdf', 'nega%d' % i, format='fixed', complevel=COMPLEVEL, complib='zlib')

def main():
    profeat()
    dragon()
    dataset()

if __name__ == '__main__':
    main()
