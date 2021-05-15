import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.metrics import roc_auc_score
import os
import sys

def load_data(dataset, nfold=5):
    print('... loading data')

    # Load the dataset
    data = np.load(dataset)['data']
    print(data.shape)
    print(data)

    partition = int(data.shape[0] / nfold)
    print(partition)
    train_set = data[:-partition]
    test_set = data[-partition:]
    return train_set, test_set

def main(dataset):
    train_set, test_set = load_data(dataset)

    X_train = train_set[:,:-1]
    y_train = train_set[:,-1]

    X_test = test_set[:,:-1]
    y_test = test_set[:,-1]

    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    model = Sequential()

    model.add(Dense(1000, input_dim=X_train.shape[1]))
    model.add(Activation('sigmoid'))

    model.add(Dense(2))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(X_train, y_train, epochs=100, batch_size=100, shuffle=True, verbose=1)

    for layer in model.layers:
        print(model.get_weights())

    score = model.evaluate(X_test, y_test, batch_size=100, verbose=1)
    auc = roc_auc_score(y_test, model.predict(X_test))
    print(auc)

if __name__ == '__main__':
    dataset = 'data/cpi.npz'
    main(dataset)
