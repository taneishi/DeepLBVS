import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import argparse
import timeit
import os

def main(args):
    start_time = timeit.default_timer()

    data = np.load(args.datafile)['data']
    data = preprocessing.minmax_scale(data)
    np.random.seed(args.random_seed)
    np.random.shuffle(data)
    train, test = train_test_split(data, test_size=args.test_size, random_state=args.random_seed)

    print('%d training, %d test samples.' % (train.shape[0], test.shape[0]))

    train_x, train_y = train[:, :-1], train[:, -1]
    test_x, test_y = test[:, :-1], test[:, -1]

    np.savez_compressed(os.path.join(os.path.dirname(__file__), 'data', 'cpi_preprocessed.npz'),
            train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)

    print('%5.2f sec for preprocessing.' % (timeit.default_timer() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default=os.path.join(os.path.dirname(__file__), 'data', 'cpi.npz'), type=str)
    parser.add_argument('--random_seed', default=123, type=int)
    parser.add_argument('--test_size', default=0.2, type=float)
    args = parser.parse_args([])

    main(args)
