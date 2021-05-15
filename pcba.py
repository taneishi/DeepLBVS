import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import os

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
        print('\r%5d/%5d' % (len(y), df[aid].notnull().sum()), end='')
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
    aid = table.iloc[0, 0]

    X, y = build(aid)
    print(X, X.shape)
    print(y, y.shape)
    np.savez_compressed('data/pcba.npz', X=X, y=y)
