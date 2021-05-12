import pandas as pd
import numpy as np
import gzip,pickle
import itertools

def mnist():
    filename = '../deep/data/mnist.pkl.gz'
    data = pickle.load(gzip.open(filename))
    mat = []
    for row in data:
        df = pd.DataFrame(row[0])
        df['label'] = row[1]
        mat.append(df)
    data = pd.concat(mat)
    np.savez('data/mnist.npz', data=data.values)

def npz():
    out = open('out1.train', 'w')
    data = np.load('data/out1.npz')['data']
    for target,row in zip(data[:,-1], data[:,:-1]):
        row = ['%d:%f' % (x[0],x[1]) for x in enumerate(row, 1)]
        out.write('%d %s\n' % (target, ' '.join(row)))
    out.close()

def libsvm():
    out = open('out1.train', 'w')
    df = pd.read_pickle('../gpcr/pkl/out1')
    for i,row in df.iterrows():
        target = row[-1]
        row = ['%d:%f' % (x[0], x[1]) for x in enumerate(row.values[:-1], 1)]
        out.write('%d %s\n' % (target, ' '.join(row)))
    out.close()

if __name__ == '__main__':
    main()
