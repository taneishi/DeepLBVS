import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import os

def predict(X, y):
    np.random.seed(123)
    cls = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(cls, X, y, cv=5, scoring='roc_auc')

    return scores.mean()

def build_table():
    df = pd.read_csv('../data/pcba.csv.gz', sep=',').set_index(['mol_id','smiles'])

    table = []
    for col in df.columns:
        negative, positive = df.groupby(col).size()
        table.append([col, df[col].notnull().sum(), negative, positive])

    table = pd.DataFrame(table, columns=['AID', 'count', 'negative', 'positive'])
    table['diff'] = np.abs(table['positive'] - table['negative'])
    table = table.sort_values(['diff', 'count'], ascending=True)

    return table
    
def build(aid, diameter=4, nbits=2048):
    df = pd.read_csv('../data/pcba.csv.gz', sep=',').set_index(['mol_id','smiles'])

    X, y = [], []
    for index, row in df.loc[df[aid].notnull(), :].iterrows():
        print('\rCompounds %5d/%5d' % (len(y), df[aid].notnull().sum()), end='')
        mol_id, smiles = index
        mol = Chem.MolFromSmiles(smiles)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, int(diameter/2), nBits=nbits, useChirality=False, useBondTypes=True, useFeatures=False)
        fp = np.asarray(fp)
        X.append(fp)
        y.append(row[aid])

    print('\n')
        
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y

if __name__ == '__main__':
    table = build_table()
    print(table)

    for aid in table.iloc[:10, 0]:
        X, y = build(aid)
        print(X.shape)

        mean_auc = predict(X, y)
        print(aid, mean_auc)

        table.loc[table['AID'] == aid, 'MeanAUC'] = mean_auc

    print(table.loc[table['MeanAUC'].notnull(), :])
