import pandas as pd
import numpy as np
from sklearn import preprocessing
import sys
import os

def load_data(dataset):
    data = pd.read_pickle(dataset).astype(np.float32)

    print 'Data scaling'
    data = preprocessing.minmax_scale(data)

    print 'Data shuffling'
    np.random.seed(123)
    data = np.random.permutation(data)

    # save
    filename = os.path.basename(dataset) + '.npz'
    np.savez_compressed(ps.path.join('/data/gpcr/npz', filename), data=data)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        load_data(dataset)
    else:
        sys.exit('Usage: python %s [pklfile]' % sys.argv[0])
