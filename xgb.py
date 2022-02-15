import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import argparse
import timeit
import os

from pcba import pcba_matrix, show_results, create_ecfp, load_ecfp

def main(args):
    np.random.seed(123)
    os.makedirs(args.log_dir, exist_ok=True)

    # dataset is provided in (aid x compounds) matrix
    df = pcba_matrix(args)
    print(df)

    show_results(args)

    # create ECFP fingerprints
    for aid in df.index:
        create_ecfp(aid, args)

    params = { 'objective': args.objective, 'num_class': args.num_class, 'eval_metric': args.eval_metric }

    for aid in df.index:
        print('\nAID %6s (%3d/%3d)' % (aid, df.index.get_loc(aid) + 1, args.limit))
        print(df.loc[df.index == aid, :'percentage'])

        X, y = load_ecfp(aid, args)

        start_time = timeit.default_timer()

        skf = StratifiedKFold(n_splits=args.n_splits)
        for fold, (train, test) in enumerate(skf.split(X, y), 1):
            d_train = xgb.DMatrix(X[train], label=y[train])
            d_test = xgb.DMatrix(X[test], label=y[test])

            cls = xgb.train(params, d_train, args.num_round)
            y_pred = cls.predict(d_test)
            auc = roc_auc_score(y[test], y_pred)

            df.loc[df.index == aid, 'AUC_%d' % (fold)] = auc

        elapsed = timeit.default_timer() - start_time

        mean_auc = df.loc[df.index == aid, 'AUC_1':'AUC_%d' % (args.n_splits)].mean(axis=1)
        df.loc[df.index == aid, 'MeanAUC'] = mean_auc

        print('%s %d-fold CV mean AUC %5.3f %5.3fsec' % ('xgb', args.n_splits, mean_auc, elapsed))

    df.loc['MeanAUC', :] = df.mean(axis=0)
    df.loc[:, 'AUC_1':] = df.loc[:, 'AUC_1':].round(4)
    df.to_csv('%s/%d_%d_results.tsv.gz' % (args.log_dir, args.diameter, args.nbits), sep='\t')
    print(df.loc[df['MeanAUC'].notnull(), :])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--dataset', default='pcba.csv.gz', type=str)
    parser.add_argument('--diameter', default=4, type=int)
    parser.add_argument('--nbits', default=1024, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--sort', default=True, action='store_true', help='Sort by positive percenrage and count of compounds')
    parser.add_argument('--limit', default=10, type=int, help='Number of AIDs to process')
    parser.add_argument('--log_dir', default='log/xgb', type=str)
    parser.add_argument('--random_seed', default=123, type=int)

    parser.add_argument('--objective', default='multi:softmax', type=str)
    parser.add_argument('--eval_metric', default='mlogloss', type=str)
    parser.add_argument('--num_class', default=2, type=int)
    parser.add_argument('--num_round', default=300, type=int)
    args = parser.parse_args()
    print(vars(args))

    main(args)
