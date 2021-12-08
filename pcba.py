import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import argparse
import timeit
import os

def predict(X, y, n_splits=5):
    np.random.seed(123)
    cls = RandomForestClassifier(n_estimators=100)

    auc = []
    skf = StratifiedKFold(n_splits=n_splits)
    for train, test in skf.split(X, y):
        cls.fit(X[train], y[train])
        y_pred = cls.predict(X[test])
        auc.append(roc_auc_score(y[test], y_pred))

    return auc

def build_ecfp(aid, diameter=4, nbits=2048):
    dirname = 'ecfp%d_%d' % (diameter, nbits)
    filename = '%s/%s.tsv.gz' % (dirname, aid)

    os.makedirs(dirname, exist_ok=True)
    if os.path.exists(filename):
        df = pd.read_csv(filename, sep='\t')
        return df.iloc[:, :-1].values, df.iloc[:, -1].values

    df = pd.read_csv('data/pcba.csv.gz', sep=',').set_index(['mol_id', 'smiles'])

    X, y = [], []
    for index, row in df.loc[df[aid].notnull(), :].iterrows():
        mol_id, smiles = index
        mol = Chem.MolFromSmiles(smiles)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, int(diameter/2), nBits=nbits, useChirality=False, useBondTypes=True, useFeatures=False)
        fp = np.asarray(fp)
        X.append(fp)
        y.append(row[aid])
        print('\rConverted compounds %6d/%6d' % (len(y), df[aid].notnull().sum()), end='')

    X = np.asarray(X)
    y = np.asarray(y)

    df = pd.DataFrame(X)
    df['outcome'] = y
    df.to_csv(filename, sep='\t', index=False)
    
    return X, y

def main(args):
    df = pd.read_csv('data/pcba.csv.gz', sep=',').set_index(['mol_id', 'smiles'])
    df = df.reset_index(drop=True).T

    for aid in df.index:
        negative, positive = df.T.groupby(aid).size()
        df.loc[aid, 'count'] = df.T[aid].notnull().sum()
        df.loc[aid, 'positive'] = positive
        df.loc[aid, 'negative'] = negative

    df = df[['count', 'negative', 'positive']]
    df['percentage'] = df['positive'] / df['count'] * 100.

    if args.sort:
        df = df.sort_values(['percentage', 'count'], ascending=False)

    if args.limit == 0:
        args.limit = df.shape[0]

    df = df.iloc[:args.limit, :]
    print(df)

    for index, aid in enumerate(df.index):
        start_time = timeit.default_timer()
        print('\nAID %s (%3d/%3d)' % (aid, index+1, args.limit))
        X, y = build_ecfp(aid)
        print('\rNumber of compounds converted %6d %5.3fsec' % (len(y), timeit.default_timer() - start_time))

        start_time = timeit.default_timer()
        auc = predict(X, y, args.n_splits)
        print('RandomForest %d-fold CV mean AUC %5.3f %5.3fsec' % (args.n_splits, np.mean(auc), timeit.default_timer() - start_time))

        for index, value in enumerate(auc, 1):
            df.loc[df.index == aid, 'AUC_%d' % (index)] = value
        df.loc[df.index == aid, 'MeanAUC'] = np.mean(auc)

        df.to_csv('results.tsv.gz', sep='\t')

    print(df.loc[df['MeanAUC'].notnull(), :])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--sort', default=False, action='store_true', help='Sort by number of compounds and positive percenrage')
    parser.add_argument('--limit', default=0, type=int, help='Number of AIDs to process')
    args = parser.parse_args()
    print(vars(args))

    main(args)
