"""
utils for custom dataset
"""
import os
import sys
import pandas as pd

import numpy

import theano
import theano.tensor as T

PATH = os.path.join(os.environ['HOME'],'deep')
INITPY = os.path.join(PATH,'code','__init__.py')
if not os.path.exists(INITPY): open(INITPY, 'w')
sys.path.insert(0, PATH)
DATASET = os.path.join(PATH,'data','mnist')

def build(dataset):
    import gzip,cPickle
    filename = dataset + '.pkl.gz'
    data = cPickle.load(gzip.open(filename))
    cat = []
    for row in data:
        df = pd.DataFrame(row[0])
        df['label'] = row[1]
        cat.append(df)
    df = pd.concat(cat)
    df.to_pickle(dataset)

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    print '... loading data'

    # Load the dataset
    df = pd.read_pickle(dataset)

    if os.path.basename(dataset) != 'mnist':
        numpy.random.seed(123)
        df = df.reindex(numpy.random.permutation(df.index))
        # purge invariance
        df = df.ix[:, df.max() != df.min()]
        # scaling [0,1]
        df.ix[:,:-1] = (df.ix[:,:-1] - df.ix[:,:-1].min()) / (df.ix[:,:-1].max() - df.ix[:,:-1].min())

    train_set = df[:-df.shape[0]/6]
    test_set = df[-df.shape[0]/6:]

    #train_set, test_set format: pandas.DataFrame(columns=[0,1,2,...,label])

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x = data_xy.ix[:,:-1]
        data_y = data_xy.ix[:,-1]
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval


if __name__ == '__main__':
    if not os.path.exists(DATASET):
        build(DATASET)
    print(load_data(DATASET))
