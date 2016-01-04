import pandas as pd
import numpy as np
import itertools

def npz():
    out = open('out1.train', 'w')
    data = np.load('data/out1.npz')['data']
    for target,row in itertools.izip(data[:,-1], data[:,:-1]):
        row = map(lambda x: '%d:%f' % (x[0],x[1]), enumerate(row, 1))
        out.write('%d %s\n' % (target, ' '.join(row)))
    out.close()

def main():
    out = open('out1.train', 'w')
    df = pd.read_pickle('../gpcr/pkl/out1')
    for i,row in df.iterrows():
        target = row[-1]
        row = map(lambda x: '%d:%f' % (x[0],x[1]), enumerate(row.values[:-1], 1))
        out.write('%d %s\n' % (target, ' '.join(row)))
    out.close()

if __name__ == '__main__':
    main()
