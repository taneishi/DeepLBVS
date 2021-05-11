import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import roc_auc_score
import os

def build(diameter=4, nbits=2048):
    df = pd.read_csv('data/pcba.csv.gz', sep=',').set_index(['mol_id','smiles'])

    table = []
    for col in df.columns:
        negative, positive = df.groupby(col).size()
        table.append([col, df[col].notnull().sum(), negative, positive])
    table = pd.DataFrame(table, columns=['AID', 'count', 'negative', 'positive'])
    table['diff'] = np.abs(table['positive'] - table['negative'])
    table = table.sort_values(['diff', 'count'], ascending=True)
    print(table)
    
    col = table.iloc[0, 0]
    print(col)
    
    X, y = [], []
    for index, row in df.loc[df[col].notnull(), :].iterrows():
        print('\r%d/%d' % (len(y), df[col].notnull().sum()), end='')
        mol_id, smiles = index
        mol = Chem.MolFromSmiles(smiles)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, int(diameter/2), nBits=nbits, useChirality=False,
            useBondTypes=True, useFeatures=False)
        fp = np.asarray(fp)
        X.append(fp)
        y.append(row[col])
        
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y

np.random.seed(123)

X, y = build()
print(X.shape, y.shape)

#dnn.validation(aid, data, layers=[2000, 100],
#        batch_size=1000, nb_epoch=100, optimizer='Adam', lr=0.0001, activation='relu', 
#        dropout=0)

def validation(X_train, y_train, X_valid, y_valid):
    print(X.shape, y.shape)

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

validation(X, y, X, y)

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

cls = RandomForestClassifier(n_estimators=100)

scores = cross_validation.cross_val_score(
            cls, data, target, cv=4, scoring='roc_auc')

print(scores.mean())
