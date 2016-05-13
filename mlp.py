# coding:utf-8
from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
import keras
import theano
import numpy as np
import pandas as pd
import timeit
import os
import sys

def validation(datafile, layers, nb_epoch, batch_size, optimizer, activation):
    print('Data loading ...')
    data = np.load(datafile)['data']
    print(str(data.shape))

    X = data[:,:-1]
    y = np_utils.to_categorical(data[:,-1], 2)

    model = Sequential()

    model.add(Dense(layers[0], input_dim=X.shape[1], init='uniform'))
    model.add(Activation(activation))

    for layer in layers[1:]:
        model.add(Dense(layer, init='uniform'))
        model.add(Activation(activation))

    # output layer
    model.add(Dense(2, init='uniform'))
    model.add(Activation('softmax'))

    model.summary()
    
    start_time = timeit.default_timer()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    earlystopping = EarlyStopping(monitor='val_loss', patience=10)

    # fitting
    history = model.fit(X, y, nb_epoch=nb_epoch, batch_size=batch_size, 
            shuffle=True, validation_split=0.2, verbose=1,
            callbacks=[earlystopping])

    end_time = timeit.default_timer()
    print('ran for %.1fs' % ((end_time - start_time)))

    # write log
    logfile = 'result/%s_%s_%d_%s_%s_%d.log' % (
            os.path.basename(datafile), 
            '_'.join(map(str, layers)), 
            batch_size, 
            optimizer,
            activation,
            nb_epoch,
            )
    df = pd.DataFrame.from_dict(history.history)
    df.to_pickle(logfile)
    print('Log file saved as %s' % logfile)

    # save model
    modelfile = 'model/%s_%s_%d_%s_%s_%d.json' % (
            os.path.basename(datafile), 
            '_'.join(map(str, layers)), 
            batch_size, 
            optimizer,
            activation,
            nb_epoch,
            )
    open(modelfile, 'w').write(model.to_json())
    model.save_weights(modelfile.replace('json','h5'), overwrite=True)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        datafile = sys.argv[1]
    else:
        sys.exit('Usage: %s [datafile]' % (sys.argv[0]))

    for dirname in ['model','result']:
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    print('Keras %s' %keras.__version__)
    print('Theano: %s' % theano.version.version)
    print('numpy: %s' % np.version.version)
    np.show_config()
    print('Python: %s' % sys.version)

    np.random.seed(123)

    optimizer = 'adam'
    activation = 'sigmoid'
    nb_epoch = 200
    for unit in [3000]:
        for batch_size in [1000]:
            for n_layers in [1]:
                validation(datafile, layers=[unit] * n_layers, batch_size=batch_size, nb_epoch=nb_epoch, optimizer=optimizer, activation=activation)
