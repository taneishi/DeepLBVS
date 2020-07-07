import numpy as np
from sklearn.preprocessing import minmax_scale
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import argparse
import timeit

def data_load():
    print('Data loading ...')
    data = np.load(args.datafile, allow_pickle=True)['data']
    data = minmax_scale(data)

    np.random.shuffle(data)
    print(data.shape)

    train, test = train_test_split(data, train_size=0.8, test_size=0.2)

    train_x, train_y = train[:,:-1], train[:,-1]
    test_x, test_y = test[:,:-1], test[:,-1]

    return train_x, train_y, test_x, test_y

def main(args):
    train_x, train_y, test_x, test_y = data_load()

    activation = 'relu'
    optimizer = Adam(lr=args.lr)

    model = Sequential()
    
    # input layer
    model.add(Dense(3000, input_dim=1974, init='uniform', name='Input'))
    model.add(Activation(activation))
    if args.dropout > 0:
        model.add(Dropout(args.dropout))

    model.add(Dense(50, init='uniform', name='Hidden'))
    model.add(Activation(activation))
    if args.dropout > 0:
        model.add(Dropout(args.dropout))

    # output layer
    model.add(Dense(1, init='uniform', name='Output'))
    model.add(Activation('sigmoid', name='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # fitting
    model.fit(train_x, train_y, nb_epoch=args.epochs, batch_size=args.batch_size, shuffle=True, 
            validation_data=(test_x, test_y), verbose=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default='data/cpi.npz')
    parser.add_argument('--modelfile', default=None, type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--batch_size', default=1500, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--random_seed', default=123, type=int)
    args = parser.parse_args()
    print(vars(args))

    np.random.seed(args.random_seed)
    main(args)
