from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import os
import sys

def load_data(dataset, nfold=5):
    data, target = load_svmlight_file(dataset)
    data = data.todense()
    data = (data - data.mean(axis=1)) / data.std(axis=1)

    target = target.reshape(data.shape[0], 1)
    data = np.concatenate((data, target), axis=1)
    data = np.asarray(data, dtype=np.float32)
    data = np.random.permutation(data)
    print data.shape
    train_set = data[:-data.shape[0] / nfold]
    test_set = data[-data.shape[0] / nfold:]
    return train_set, test_set

def validation(dataset):
    train_set, test_set = load_data(dataset)

    X_train = train_set[:,:-1]
    y_train = train_set[:,-1]

    X_test = test_set[:,:-1]
    y_test = test_set[:,-1]

    model = Sequential()

    model.add(Dense(1000, input_dim=X_train.shape[1], init='uniform'))
    model.add(Activation('sigmoid'))

    model.add(Dense(1000, init='uniform'))
    model.add(Activation('sigmoid'))

    model.add(Dense(1, init='uniform'))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, class_mode='binary')

    model.fit(X_train, y_train, nb_epoch=100, batch_size=10)
    score = model.evaluate(X_test, y_test, batch_size=10)

    for layer in model.layers:
        print model.get_weights()

    auc = roc_auc_score(y_test, model.predict_proba(X_test))
    out = open('result.log', 'a')
    out.write('%s\t%.3f\n' % (dataset,auc))
    out.close()

if __name__ == '__main__':
    dirname = 'ecfp'
    for dataset in sorted(os.listdir(dirname)):
        if os.path.isdir(os.path.join(dirname, dataset)): continue
        print dataset
        validation(os.path.join(dirname, dataset))
