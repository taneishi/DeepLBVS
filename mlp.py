import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import argparse
import timeit
import os

from pcba import pcba_matrix, create_ecfp, load_ecfp

class MLP(nn.Module):
    def __init__(self, input_dim=1974, dropout=0.1):
        # 2000, 100, 2, relu, adam, lr=0.0001, dropout=0
        # 1000, 1000, 2, sigmoid, sgd, lr=0.01, momentum=0.9, mse
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, 3000)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(3000, 50)
        self.dropout2 = nn.Dropout(dropout)
        self.output_layer = nn.Linear(50, 2)
        
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.output_layer(x))
        return x

def main(args):
    np.random.seed(args.random_seed)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using %s device.' % (device))

    # dataset is provided in (aid x compounds) matrix
    df = pcba_matrix(args)
    print(df)

    # create ECFP fingerprints
    for aid in df.index:
        create_ecfp(aid, args)

    for aid in df.index:
        print('\nAID %6s (%3d/%3d)' % (aid, df.index.get_loc(aid) + 1, args.limit))
        print(df.loc[df.index == aid, :'percentage'])

        X, y = load_ecfp(aid, args)

        start_time = timeit.default_timer()

        skf = StratifiedKFold(n_splits=args.n_splits)
        for fold, (train, test) in enumerate(skf.split(X, y), 1):
            # create torch tensor from numpy array
            train_x = torch.FloatTensor(X[train]).to(device)
            train_y = torch.LongTensor(y[train]).to(device)
            train = torch.utils.data.TensorDataset(train_x, train_y)
            train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True)

            net = MLP(input_dim=args.nbits, dropout=args.dropout)
            net = net.to(device)
        
            if args.modelfile:
                net.load_state_dict(torch.load(args.modelfile))

            net.train()

            # define our optimizer and loss function
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            loss_func = F.cross_entropy

            for epoch in range(args.epochs):
                epoch_start = timeit.default_timer()

                train_loss = 0

                for index, (data, label) in enumerate(train_dataloader, 1):
                    optimizer.zero_grad()
                    output = net(data)
                    loss = loss_func(output, label, reduction='mean')
                    train_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                print('\rfold %d epoch %4d batch %4d/%4d' % (fold, epoch, index, len(train_dataloader)), end='')
                print(' train_loss %5.3f %5.3fsec' % (train_loss / index, timeit.default_timer() - epoch_start), end='')

            test_x = torch.FloatTensor(X[test]).to(device)
            test_y = torch.LongTensor(y[test]).to(device)
            test = torch.utils.data.TensorDataset(test_x, test_y)
            test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size)

            net.eval()

            test_loss = 0
            y_score, y_true = [], []
            for index, (data, label) in enumerate(test_dataloader, 1):
                with torch.no_grad():
                    output = net(data)
                loss = loss_func(output, label, reduction='mean')
                test_loss += loss.item()
                y_score.append(output.cpu())
                y_true.append(label.cpu())

            y_score = np.concatenate(y_score)
            y_pred = [np.argmax(x) for x in y_score]
            y_true = np.concatenate(y_true)
            confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=[1,0]).flatten()

            acc = metrics.accuracy_score(y_true, y_pred)
            auc = metrics.roc_auc_score(y_true, y_score[:,1])
            prec = metrics.precision_score(y_true, y_pred)
            recall = metrics.recall_score(y_true, y_pred)

            df.loc[df.index == aid, 'AUC_%d' % (fold)] = auc

            print(' %s test_loss %5.3f test_auc %5.3F test_prec %5.3f test_recall %5.3f' % (
                confusion_matrix, test_loss / index, auc, prec, recall))

            torch.save(net.state_dict(), os.path.join(args.model_dir, 'aid%s_fold_%d.pth' % (aid, fold)))

        elapsed = timeit.default_timer() - start_time

        mean_auc = df.loc[df.index == aid, 'AUC_1':'AUC_%d' % (args.n_splits)].mean(axis=1)
        df.loc[df.index == aid, 'MeanAUC'] = mean_auc

        print('MLP %d-fold CV mean AUC %5.3f %5.3fsec' % (args.n_splits, mean_auc, elapsed))

    df.loc['MeanAUC', :] = df.mean(axis=0)
    df.loc[:, 'AUC_1':] = df.loc[:, 'AUC_1':].round(4)
    df.to_csv('%s/%d_%d_results.tsv.gz' % (args.log_dir, args.diameter, args.nbits), sep='\t')
    print(df.loc[df['MeanAUC'].notnull(), :])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--dataset', default='pcba.csv.gz', type=str)
    parser.add_argument('--diameter', default=4, type=int)
    parser.add_argument('--nbits', default=1024, type=int)
    parser.add_argument('--n_splits', default=5, type=int, help='a number of folds of cross validation')
    parser.add_argument('--sort', default=True, action='store_true', help='Sort by positive percenrage and count of compounds')
    parser.add_argument('--limit', default=10, type=int, help='Number of AIDs to process')
    parser.add_argument('--log_dir', default='log/mlp', type=str)
    parser.add_argument('--random_seed', default=123, type=int)
    parser.add_argument('--model_dir', default='model', type=str)
    parser.add_argument('--modelfile', default=None, type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0., type=float)
    args = parser.parse_args()
    print(vars(args))

    main(args)
