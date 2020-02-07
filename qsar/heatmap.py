import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import os
import sys

name2aid = lambda x: int(os.path.basename(x).split('_')[0].split('-')[1])

def main():
    index = sorted(sys.argv[1:], key=name2aid)
    mat = []
    for filename in index:
        if filename.endswith('roc'):
            continue
        aid = name2aid(filename)
        df = pd.read_pickle(filename)
        df['AID'] = aid
        #fpr, tpr, threashold = metrics.roc_curve(df.ix[cond, 'label'], df.ix[cond, 0])
        #print(metrics.auc(fpr,tpr))
        mat.append(df[['AID', 'val_acc', 'auc']])

    mat = pd.concat(mat)
    del mat.index.name
    mat = mat.reset_index()

    for col in ['val_acc', 'auc']:
        df = pd.pivot_table(mat, index='AID', columns='index', values=col)

        print(df.shape)
        plt.figure(figsize=(12,8))
        sns.heatmap(df, xticklabels=10, yticklabels=5, vmin=.6, vmax=1., cmap='inferno')
        plt.yticks(rotation=0)
        plt.xlabel('Epoch')
        plt.ylabel('PubChem AID (n=%d)' % df.shape[0])
        plt.tight_layout()
        #plt.show()
        plt.savefig('%s.png' % col)
        plt.close()

    print(df, df.max(axis=1).describe())
    #plt.figure(figsize=(12,8))

    print(df.max(axis=1).to_frame())
    plt.figure(figsize=(4,6))
    df.max(axis=1).to_frame().rename(columns={0:'AUC'}).boxplot(return_type='axes')
    plt.show()


if __name__ == '__main__':
    main()
