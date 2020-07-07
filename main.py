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

def main(args):
    print('Data loading ...')
    data = np.load(args.datafile, allow_pickle=True)['data']
    data = minmax_scale(data)
    print(data)

    np.random.shuffle(data)
    print(data.shape)

    train, test = train_test_split(data, train_size=0.8, test_size=0.2)

    train_x, train_y = train[:,:-1], train[:,-1]
    test_x, test_y = test[:,:-1], test[:,-1]

    lr = 0.0001
    epochs = 300
    dropout = 0.1
    batch_size = 1500
    activation = 'relu'

    optimizer = Adam(lr=lr)

    model = Sequential()
    
    # input layer
    model.add(Dense(3000, input_dim=1974, init='uniform', name='Input'))
    model.add(Activation(activation))
    if dropout > 0:
        model.add(Dropout(dropout))

    model.add(Dense(50, init='uniform', name='Hidden'))
    model.add(Activation(activation))
    if dropout > 0:
        model.add(Dropout(dropout))

    # output layer
    model.add(Dense(1, init='uniform', name='Output'))
    model.add(Activation('sigmoid', name='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    # fitting
    model.fit(train_x, train_y, nb_epoch=epochs, batch_size=batch_size, shuffle=True, 
            validation_data=(test_x, test_y), verbose=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', default='cpi.npz')
    args = parser.parse_args()

    np.random.seed(123)
    main(args)
