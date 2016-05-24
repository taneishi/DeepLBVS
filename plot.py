# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys
import os

def ascend(df, col):
    data = []
    for i,row in df.iterrows():
        cond = df['epoch'] < row['epoch']
        if len(data) == 0 or row[col] > df.ix[cond, col].max():
            data.append(row[col])
        else:
            data.append(np.nan)
    return data

def main(nb_epoch=500):
    plt.figure(figsize=(12,8))

    logs = []
    for filename in sys.argv[1:]:
        label = os.path.basename(filename).replace('.log','')
        datafile = label.split('_')[0]

        if False and os.path.exists('data/%s' % datafile):
            if datafile.endswith('npz'):
                data = np.load('data/%s' % datafile)['data'] 
            else:
                data = pd.read_pickle('data/%s' % datafile).values
            npos = (data[:,-1] == 1).sum()
            nneg = (data[:,-1] == 0).sum()
            rate = '(P:N=%d:%d)' % (npos, nneg)
            label = '%s %s' % (label,rate)

        df = pd.read_pickle(filename)
        if 0 in df.columns: # old
            df.columns = ['epoch','val_error']
            df['val_acc'] = 1.0 - df['val_error']
        else: # new
            df['epoch'] = range(1, df.shape[0]+1)

        df = df[df['epoch'] <= nb_epoch]

        if 'acc' in df.columns:
            df['acc'] = ascend(df, col='acc')
        df['val_acc'] = ascend(df, col='val_acc') 

        logs.append([df, label])

    logs = sorted(logs, key=lambda x: x[0]['val_acc'].max(), reverse=True)

    for df,label in logs[:10]:
        val = df[['epoch','val_acc']].dropna()
        minutes = '%d min' % (df['time'].max() / 60.) if 'time' in df.columns else ''
        line, = plt.plot(val['epoch'], val['val_acc'] * 100.0, '^-',
                label='%s_val %.1f at %d (%s)' % (
                    label,val['val_acc'].max()*100.0, val['epoch'].max(), minutes,
                    )) 
        if True:
            if 'acc' in df.columns:
                acc = df[['epoch','acc']].dropna()
                plt.plot(acc['epoch'], acc['acc'] * 100.0,
                        label='%s %.1f at %d' % (
                            label,acc['acc'].max()*100.0, acc['epoch'].max()
                            ), color=line.get_color()) 

    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.xlim(0,nb_epoch)
    plt.ylim(50.,100.)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('%s [filename]' % sys.argv[0])
    main(nb_epoch=200)
