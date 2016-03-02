import numpy as np
from sklearn import preprocessing
import gzip
import sys
import os
import time

def load_data(dataset):
    # load
    data = []
    for l in gzip.open(dataset):
        array = np.asarray(l.strip().split('\t'), dtype=np.float32)
        data.append(array)
    data = np.asmatrix(data)

    # scaling
    data = preprocessing.minmax_scale(data)

    # save
    filename = os.path.basename(dataset).replace('tsv.gz','npz')
    np.savez_compressed(os.path.join('/data/gpcr', filename), data=data)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
        load_data(dataset)
    else:
        sys.exit('Usage: python %s [tsvfile]' % sys.argv[0])
