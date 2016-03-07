import pandas as pd
import numpy as np
import gzip
import os
import sys

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

def main(n=None):
    np.random.seed(123)

    store = pd.HDFStore('cpi.hdf')
    dragon = store['dragon']
    profeat = store['profeat']
    posi = store['posi']
    nega = store['nega1']
    store.close()
    del store

    protname = pd.read_csv('protname_out.txt', sep='\t', header=None)
    id2name = protname.set_index(3)[0].to_dict()

    df = pd.concat([posi,nega])
    df = df[df[0].isin(dragon.index)]
    df = df[df[1].isin(profeat.index)]

    df = df.reset_index(drop=True)
    df = df.reindex(np.random.permutation(df.index))
    if n is not None:
        df = df.iloc[:n, :]

    print df.shape

    if n < 1e6:
        filename = '%dK.tsv.gz' % (int(n / 1e3))
    else:
        filename = '%dM.tsv.gz' % (int(n / 1e6))
    out = gzip.open(filename, 'w')
    for i,row in df.iterrows():
        out.write('%s\t%s\t%d\n' % (
                '\t'.join(map(str, dragon.ix[row[0],:])),
                '\t'.join(map(str, profeat.ix[row[1],:])),
                row[2]))
    out.close()

if __name__ == '__main__':
    #profeat()
    #dragon()
    #dataset()
    main(n=5e5)
