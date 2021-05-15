import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import os

#dnn.validation(aid, data, layers=[2000, 100],
#        batch_size=1000, nb_epoch=100, optimizer='Adam', lr=0.0001, activation='relu', 
#        dropout=0)

def validation(X_train, y_train, X_valid, y_valid):
    model = Sequential()

    model.add(Dense(1000, input_dim=X_train.shape[1], kernel_initializer='uniform'))
    model.add(Activation('sigmoid'))

    model.add(Dense(1000, kernel_initializer='uniform'))
    model.add(Activation('sigmoid'))

    model.add(Dense(1, kernel_initializer='uniform'))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    model.fit(X_train, y_train, epochs=100, batch_size=100)
    score = model.evaluate(X_valid, y_valid, batch_size=100)

    for i, layer in enumerate(model.layers):
        print(i, len(layer.get_weights()))

    auc = roc_auc_score(y_valid, model.predict_proba(X_valid))
    
    print('AUC\t%.3f\n' % (auc))

def main():
    np.random.seed(123)
    
    data = np.load('data/pcba.npz')
    X = data['X']
    y = data['y']

    cls = RandomForestClassifier(n_estimators=100)

    scores = cross_val_score(cls, X, y, cv=4, scoring='roc_auc')
    print(scores.mean())

    #validation(X, y, X, y)

if __name__ == '__main__':
    main()
