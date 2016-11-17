from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from sklearn.cross_validation import StratifiedKFold, KFold
import keras
import theano
import numpy as np
import pandas as pd
import timeit
import os
import sys

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.timehistory = []

    def on_epoch_end(self, batch, logs={}):
        print(timeit.default_timer())
        self.timehistory.append(timeit.default_timer())
        logs['time'] = timeit.default_timer()

def setup():
    for dirname in ['model','result']:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

def versions():
    versions = (('Keras',keras.__version__),
            ('Theano', theano.version.version),
            ('numpy', np.version.version),
            ('Python', sys.version))
    return versions

def show_version():
    for version in versions():
        print(' '.join(version))

def validation(taskname, data, layers, nb_epoch, batch_size, optimizer, lr, activation, dropout, patience, count):
    X = data[:,:-1]
    y = np_utils.to_categorical(data[:,-1], len(np.unique(data[:,-1])))

    optimizer = optimizer(lr=lr)

    start_time = timeit.default_timer()

    skf = StratifiedKFold(y[:,0], n_folds=5, shuffle=True)
    log = [] 
    proba = []
    for i, (train, test) in enumerate(skf, 1):
        callbacks = []
        if patience > 0:
            earlystopping = EarlyStopping(monitor='val_loss', patience=patience)
            callbacks.append(earlystopping)
        if count == 0:
            break
        timehistory = TimeHistory()
        callbacks.append(timehistory)

        model = Sequential()

        # input layer
        model.add(Dense(layers[0], input_dim=X.shape[1], init='uniform'))
        model.add(Activation(activation))

        for layer in layers[1:]:
            model.add(Dense(layer, init='uniform'))
            model.add(Activation(activation))
            if dropout > 0:
                model.add(Dropout(dropout))

        # output layer
        model.add(Dense(y.shape[1], init='uniform'))
        model.add(Activation('softmax'))

        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # fitting
        history = model.fit(X[train], y[train], nb_epoch=nb_epoch, batch_size=batch_size, 
                shuffle=True, validation_data=(X[test], y[test]), verbose=1,
                callbacks=callbacks)

        df = pd.DataFrame(model.predict_proba(X[test]))
        df['label'] = y[test,0]
        df['fold'] = i
        proba.append(df)

        df = pd.DataFrame.from_dict(history.history)
        df['time'] = df['time'] - df['time'].min()
        df['fold'] = i
        log.append(df)

        count -= 1

    end_time = timeit.default_timer()
    print('ran for %.1fs' % ((end_time - start_time)))

    basename = '%s_%s_%d_%s_%f_%s_%.1f_%d' % (
            taskname, '_'.join(map(str, layers)), 
            batch_size, str(optimizer).split(' ')[0].split('.')[-1].lower(), lr,
            activation, dropout, nb_epoch)

    # write score
    df = pd.concat(proba)
    df.to_pickle('result/%s.roc' % basename)

    # write log
    df = pd.concat(log)
    df.index.name = versions()

    df.to_pickle('result/%s.log' % basename)
    print('Log file saved as result/%s.log' % basename)

    # save model
    modelfile = 'model/%s.json' % basename
    open(modelfile, 'w').write(model.to_json())
    model.save_weights(modelfile.replace('json','h5'), overwrite=True)

if __name__ == '__main__':
    show_version()
