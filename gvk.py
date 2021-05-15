import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
import os

def main(family='gpcr'):
    data_dir = '../data/gvk/%s' % (family)

    dfs = []
    for split in ['train', 'test']:
        pair = pd.read_csv(os.path.join(data_dir, 'cgbvs-%s5000posneg_trial1_%s.vec.txt.gz' % (family, split)),
                sep=' ', index_col=0, header=None)

        df = pd.read_csv(os.path.join(data_dir, 'cpis_%s_5000posneg_%s.trial_1.txt.gz' % (family, split)), header=None)
        df[0] = df[0].str.split('=').map(lambda x: x[1])
        df[1] = df[1].str.split('=').map(lambda x: x[1])
        df[2] = df[2].str.split('=').map(lambda x: x[1])
        df['label'] = df[2].str.startswith('interaction').astype(int)
        df.index = df[0] + '--' + df[1]
        df = pd.merge(pair, df['label'], left_index=True, right_index=True)
        df['train'] = split == 'train'

        dfs.append(df)

    train = dfs[0]
    test = dfs[1]
    df = pd.concat(dfs)
    print(df)

    train = preprocessing.minmax_scale(train, axis=0)
    test = preprocessing.minmax_scale(test, axis=0)

    data = np.load(os.path.join(data_dir, '%s10k_train.npz' % (family)))
    X_train = data['X_train']
    y_train = data['y_train']

    data = np.load(os.path.join(data_dir, '%s10k_valid.npz' % (family)))
    X_valid = data['X_valid']
    y_valid = data['y_valid']

    data = np.load(os.path.join(data_dir, '%s10k_test.npz' % (family)))
    X_test = data['X_test']
    y_test = data['y_test']

    print(X_train.shape, y_train.shape)
    print(X_train, y_train)

    print(X_valid.shape, y_valid.shape)
    print(X_valid, y_valid)

    print(X_test.shape, y_test.shape)
    print(X_test, y_test)

if __name__ == '__main__':
    main(family='gpcr')
    main(family='kinase')
