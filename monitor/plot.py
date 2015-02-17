# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
import datetime
import sys
import os

def main():
    logfile = os.path.join(os.environ['HOME'], '.local/share/snmp.log')
    df = pd.read_csv(logfile, sep=' ', header=None, 
            names=['OID', 'sign', 'type', 'value'])

    JST = datetime.timedelta(0,9*60*60)
    date, time = None, None
    mat = []
    for i,row in df.iterrows():
        if row['OID'].endswith('1.3.2.2.3.4.8.2.1.0'):
            date = row['value']
        elif row['OID'].endswith('1.3.2.2.3.4.8.2.2.0'):
            time = row['value']
        else:
            mat.append([parse('%s %s' % (date,time)) + JST,
                row['OID'].split('.')[-1],row['value']])

    df = pd.DataFrame(mat, columns=['time','outlet','value'])
    df = df.pivot(index='time', columns='outlet', values='value')
    df = df.astype(float)

    if len(sys.argv) > 1:
        df = df.ix[:, df.columns == sys.argv[1]]

    df = df.ix[:, (df > 0).sum() > 0]
    delta = (df.index.max() - df.index) / np.timedelta64(1,'D')
    df = df[delta <= 10]

    df.plot(style='.--', figsize=(8,6), legend=False)
    #plt.legend(loc='upper left',framealpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Ampere')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
