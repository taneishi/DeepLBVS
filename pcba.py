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

def pcba_matrix(args):
    matrix_file = '%s/%s' % (args.data_dir, args.dataset)
    df = pd.read_csv(matrix_file, sep=',').set_index(['mol_id', 'smiles'])
    df = df.reset_index(drop=True).T

    for aid in df.index:
        negative, positive = df.T.groupby(aid).size()
        df.loc[aid, 'count'] = df.T[aid].notnull().sum()
        df.loc[aid, 'positive'] = positive
        df.loc[aid, 'negative'] = negative

    df = df[['count', 'negative', 'positive']].astype(int)
    df['percentage'] = np.round(df['positive'] / df['count'] * 100., 2)

    if args.sort:
        df = df.sort_values(['percentage', 'count'], ascending=False)

    df = df.iloc[:args.limit, :]
    return df

def create_ecfp(aid, args):
    dirname = 'ecfp/%d_%d' % (args.diameter, args.nbits)
    filename = '%s/%s.tsv.gz' % (dirname, aid)

    os.makedirs(dirname, exist_ok=True)
    if os.path.exists(filename):
        return

    matrix_file = '%s/%s' % (args.data_dir, args.dataset)
    df = pd.read_csv(matrix_file, sep=',').set_index(['mol_id', 'smiles'])

    start_time = timeit.default_timer()
    X, y = [], []
    for index, row in df.loc[df[aid].notnull(), :].iterrows():
        mol_id, smiles = index
        mol = Chem.MolFromSmiles(smiles)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                mol, int(args.diameter/2), nBits=args.nbits, useChirality=False, useBondTypes=True, useFeatures=False)
        fp = np.asarray(fp)
        X.append(fp)
        y.append(row[aid])
        print('\rConverted compounds %6d/%6d' % (len(y), df[aid].notnull().sum()), end='')

    print('\nNumber of compounds converted %6d %5.3fsec' % (len(y), timeit.default_timer() - start_time))

    X = np.asarray(X)
    y = np.asarray(y)

    df = pd.DataFrame(X)
    df['outcome'] = y
    df.to_csv(filename, sep='\t', index=False)
    
def load_ecfp(aid, args):
    dirname = 'ecfp/%d_%d' % (args.diameter, args.nbits)
    filename = '%s/%s.tsv.gz' % (dirname, aid)

    df = pd.read_csv(filename, sep='\t')

    X = df.iloc[:, :-1].values
    y = df['outcome'].values
    
    return X, y

def show_results(args):
    results = dict()

    diameter_list, nbits_list = [], []
    for filename in os.listdir(args.log_dir):
        diameter, nbits, basename = filename.split('_')
        diameter_list.append(int(diameter))
        nbits_list.append(int(nbits))

    for diameter in sorted(diameter_list):
        for nbits in sorted(nbits_list):
            filename = '%s/%d_%d_results.tsv.gz' % (args.log_dir, diameter, nbits)
            if os.path.exists(filename):
                df = pd.read_csv(filename, sep='\t', index_col=0)
                results.update({'%d_%d' % (diameter, nbits): df['MeanAUC']})
        
    df = pd.DataFrame.from_dict(results)
    df.loc['MeanAUC', :] = df.mean(axis=0)
    print(df)

def main(args):
    np.random.seed(123)
    os.makedirs(args.log_dir, exist_ok=True)

    # dataset is provided in (aid x compounds) matrix
    df = pcba_matrix(args)
    print(df)

    # create ECFP fingerprints
    for aid in df.index:
        create_ecfp(aid, args)

    show_results(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--dataset', default='pcba.csv.gz', type=str)
    parser.add_argument('--diameter', default=4, type=int)
    parser.add_argument('--nbits', default=1024, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--sort', default=True, action='store_true', help='Sort by positive percenrage and count of compounds')
    parser.add_argument('--limit', default=10, type=int, help='Number of AIDs to process')
    parser.add_argument('--log_dir', default='log', type=str)
    parser.add_argument('--random_seed', default=123, type=int)
    args = parser.parse_args()
    print(vars(args))

    main(args)
