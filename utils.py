# utils for custom dataset
import os
import sys
import pandas as pd

import numpy

import theano
import theano.tensor as T

PATH = os.path.join(os.environ['HOME'],'deep')
sys.path.insert(0, PATH)
DATASET = os.path.join(PATH,'data','mnist')

def build(dataset):
    import gzip,cPickle
    filename = dataset + '.pkl.gz'
    data = cPickle.load(gzip.open(filename))
    label = []
    for row in data:
        df = pd.DataFrame(row[0])
        df['label'] = row[1]
        label.append(df)
    df = pd.concat(label)
    df.to_pickle(dataset)

def result(classifier, y):
    # check if y has same dimension of y_pred
    if y.ndim != classifier.y_pred.ndim:
        raise TypeError(
            'y should have the same shape as self.y_pred',
            ('y', y.type, 'y_pred', classifier.y_pred.type)
        )
    # check if y is of the correct datatype
    if y.dtype.startswith('int'):
        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return (classifier.p_y_given_x, classifier.y_pred, y)
    else:
        raise NotImplementedError()

def load_data(dataset, nfold=5):
    #############
    # LOAD DATA #
    #############

    print '... loading data'

    # Load the dataset
    df = numpy.load(dataset)['data']

    train_set = df[:-df.shape[0] / nfold]
    test_set = df[-df.shape[0] / nfold:]

    def shared_dataset(data_xy, borrow=True):
        data_x = data_xy[:,:-1]
        data_y = data_xy[:,-1]
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    if not os.path.exists(DATASET):
        build(DATASET)
    print(load_data(DATASET))
