from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
import sys

def validation(dataset, W):
    data, target = load_svmlight_file(dataset)
    data = data.todense()
    data = preprocessing.minmax_scale(data)
    print(data.shape)

    X = data[:,:-1]
    y = data[:,-1]

    X_train = train_set[:,:-1]
    y_train = train_set[:,-1]

    X_test = test_set[:,:-1]
    y_test = test_set[:,-1]

    model = Sequential()

    if W[0] is None:
        model.add(Dense(1000, input_dim=X_train.shape[1], init='uniform'))
    else:
        model.add(Dense(1000, input_dim=X_train.shape[1], weights=W[0]))
    model.add(Activation('sigmoid'))

    if W[1] is None:
        model.add(Dense(1000, init='uniform'))
    else:
        model.add(Dense(1000, weights=W[1]))
    model.add(Activation('sigmoid'))

    model.add(Dense(1, init='uniform'))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, class_mode='binary')

    model.fit(X_train, y_train, nb_epoch=100, batch_size=10)
    score = model.evaluate(X_test, y_test, batch_size=10)

    for i,layer in enumerate(model.layers[0:len(W)*2:2]):
        print(i,len(layer.get_weights()))
        W[i] = layer.get_weights()

    auc = roc_auc_score(y_test, model.predict_proba(X_test))
    out = open('multi.log', 'a')
    out.write('%s\t%.3f\n' % (dataset,auc))
    out.close()
    return W

if __name__ == '__main__':
    W = [None, None]
    dirname = 'dragon'
    for dataset in sorted(os.listdir(dirname)):
        if os.path.isdir(os.path.join(dirname, dataset)): continue
        print(dataset)
        W = validation(os.path.join(dirname, dataset), W)
