# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import sys
import os

def descend(df):
    data = []
    for i,row in df.iterrows():
        if row['score'] <= df.ix[df['epoch'] <= row['epoch'], 'score'].min():
            data.append((row['epoch'],row['score']))
    return pd.DataFrame(data, columns=['epoch','score'])

def main():
    if len(sys.argv) < 2:
        sys.exit('%s [filename]' % sys.argv[0])

    plt.figure(figsize=(8,6),dpi=100)

    for filename in sys.argv[1:]:
        label = os.path.basename(filename).replace('.log','')
        datafile = label.split('_')[0]
        if os.path.exists('data/%s' % datafile):
            if datafile.endswith('npz'):
                data = np.load('data/%s' % datafile)['data'] 
            else:
                data = pd.read_pickle('data/%s' % datafile).values
            npos = (data[:,-1] == 1).sum()
            nneg = (data[:,-1] == 0).sum()
            rate = '(P:N=%d:%d)' % (npos, nneg)
            label = '%s %s' % (label,rate)

        df = pd.read_pickle(filename)
        df.columns = ['epoch','score']
        df = descend(df) 

        plt.plot(df['epoch'], df['score'] * 100.0,
                label='%s %.1f' % (
                    label,df['score'].min()*100.0
                    )) 

    plt.ylabel('Error rate (%)')
    plt.xlabel('Epochs')
    plt.xlim(0,1000)
    plt.ylim(0,50)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
