from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
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

def load_data_dragon(dataset, nfold=5):
    data, target = load_svmlight_file(dataset)
    data = data.todense()
    data[:,:-1] = preprocessing.normalize(data[:,:-1])
    data = np.asarray(data, dtype=np.float32)
    data = np.random.permutation(data)
    print data.shape
    train_set = data[:-data.shape[0] / nfold]
    train_set = (train_set[:,:-1], train_set[:,-1])
    test_set = data[-data.shape[0] / nfold:]
    test_set = (test_set[:,:-1], test_set[:,-1])
    return train_set, test_set

def load_data_ecfp(dataset, nfold=5):
    data = np.load(dataset)['data']
    data = np.asarray(data, dtype=np.int8)
    print data.shape

    pos = data[data[:,-1] == 1, :]
    neg = data[data[:,-1] == 0, :]
    nsamples = min(pos.shape[0], neg.shape[0])

    # undersampleing
    neg = np.random.permutation(neg)
    neg = neg[:nsamples, :]

    data = np.concatenate([pos,neg])
    data = np.random.permutation(data)

    train_set = data[:-data.shape[0] / nfold]
    train_set = (train_set[:,:-1], np_utils.to_categorical(train_set[:,-1], 2))
    test_set = data[-data.shape[0] / nfold:]
    test_set = (test_set[:,:-1], np_utils.to_categorical(test_set[:,-1], 2))
    return train_set, test_set

def validation(dataset, nb_epoch=100):
    train_set, test_set = load_data_ecfp(dataset)

    X_train, y_train = train_set
    X_test, y_test = test_set

    model = Sequential()

    model.add(Dense(1000, input_dim=X_train.shape[1], init='uniform'))
    model.add(Activation('sigmoid'))

    model.add(Dense(1000, init='uniform'))
    model.add(Activation('sigmoid'))

    model.add(Dense(2, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    history = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=10, shuffle=True,
            validation_data=(X_test, y_test),
            show_accuracy=True, verbose=1)

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
    #plt.show()
    plt.savefig('/data/results/ecfp3000/%s.png' % basename)

if __name__ == '__main__':
    dirname = '/data/ecfp3000'
    for dataset in sorted(os.listdir(dirname)):
        if os.path.isdir(os.path.join(dirname, dataset)): continue
        print dataset
        validation(os.path.join(dirname, dataset))
