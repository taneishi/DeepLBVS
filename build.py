import pandas as pd
import numpy as np
import gzip

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
    main(n=5e5)
