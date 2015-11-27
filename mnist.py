# coding:utf-8
import pandas as pd
import numpy as np
import gzip,cPickle

def main():
    filename = '../deep/data/mnist.pkl.gz'
    data = cPickle.load(gzip.open(filename))
    mat = []
    for row in data:
        df = pd.DataFrame(row[0])
        df['label'] = row[1]
        mat.append(df)
    data = pd.concat(mat)
    np.savez('data/mnist.npz', data=data.values)

if __name__ == '__main__':
    main()
