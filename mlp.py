from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from sklearn.datasets import load_svmlight_file
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import os
import sys

def load_data(dataset):
    df = pd.read_pickle(dataset)
    print df.shape
    df = df.dropna()
    print df.shape
    data = df.values
    data = preprocessing.minmax_scale(data)
    return data[:,:-1], np_utils.to_categorical(data[:,-1], 2)

def prediction():
    # prediction
    y_pred = model.predict_proba(X_test)
    auc = metrics.roc_auc_score(y_test, y_pred)

    for layer in model.layers:
        pass
        #print model.get_weights()

    basename = os.path.basename(dataset).replace('.npz','')
    plt.figure(figsize=(8,8))
    fpr, tpr, threshold = metrics.roc_curve(y_test[:,1], model.predict_proba(X_test)[:,1])
    pos = y_test[:,1].sum()
    neg = y_test[:,0].sum()

    #plt.plot(fpr, tpr, label='%s AUC=%0.3f\nP=%d N=%d' % (basename, auc, pos, neg))
    #plt.plot(fpr, tpr, label='%s P=%d N=%d' % (basename, pos, neg))
    epochs = np.arange(nb_epoch) / float(nb_epoch)
    plt.plot(epochs, history.history['acc'], label='acc %.3f' % history.history['acc'][-1])
    plt.plot(epochs, history.history['val_acc'], label='val_acc %.3f' % history.history['val_acc'][-1])

    plt.title('%s, P:N = %d:%d,  %d epochs' % (basename, pos, neg, nb_epoch))
    plt.ylim(0,1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

def validation(dataset, descriptor, nb_epoch=100, layers=[1000,1000], batch_size=10, optimizer=['Adam','adam'], activation='sigmoid'):
    X, y = load_data(dataset)

    model = Sequential()

    model.add(Dense(layers[0], input_dim=X.shape[1], init='uniform'))
    model.add(Activation(activation))

    for units in layers[1:]:
        model.add(Dense(units, init='uniform'))
        model.add(Activation(activation))

    model.add(Dense(2, init='uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer[1])

    history = model.fit(X, y, nb_epoch=nb_epoch, 
            batch_size=batch_size, shuffle=True, validation_split=0.25,
            show_accuracy=True, verbose=1)

    df = pd.DataFrame.from_dict(history.history)
    df.to_pickle('log/%s_%s_%s_%d_%s_%s_%d.log' % (
        os.path.basename(dataset),
        descriptor,
        '_'.join(map(str,layers)),
        batch_size,
        optimizer[0],
        activation,
        nb_epoch,
        ))

if __name__ == '__main__':
    #descriptor = 'ecfp3000'
    descriptor = 'dragon'
    #optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = ['Adam', 'adam']
    for n_layer in [1,2]:
        for units in [500,1000]:
            for dataset in sorted(os.listdir(descriptor)):
                if os.path.isdir(os.path.join(descriptor, dataset)): continue
                print dataset
                validation(os.path.join(descriptor, dataset), descriptor=descriptor, 
                        nb_epoch=200, layers=[units]*n_layer, batch_size=10, 
                        optimizer=optimizer, activation='relu')
