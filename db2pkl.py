import pandas as pd
import numpy as np
import pandas.io.sql as psql
from sklearn import datasets
import sqlite3
import sys

def main():
    db = sqlite3.connect('/data/db/std_gpcr.db')
    tables = pd.read_sql_query("select name from sqlite_master where type = 'table';", db)
    print(tables)

    posi = pd.read_sql_query('select * from posi', db)
    nega = pd.read_sql_query('select * from nega', db)

    print(posi.groupby('id').size())
    print(posi[['chem','prot']].shape)
    print('Unordered')
    print(posi[posi['id'] == 0].head())
    print('Ordered')
    print(posi[posi['id'] == 1].head())

    print(posi[['chem','prot']].drop_duplicates().shape)
    print(nega.groupby('id').size())
    print(nega[['chem','prot']].drop_duplicates().shape)
    for i in nega['id'].unique():
        print(nega[nega['id'] == i].head())

    print(posi['prot'].nunique())
    print(posi['chem'].nunique())

    print(nega['prot'].nunique())
    print(nega['chem'].nunique())

    prot = pd.read_sql_query('select * from ptable', db)
    prot = prot.rename(columns={'name':'prot'})
    print(prot.head())
    chem = pd.read_sql_query('select * from ctable', db)
    chem = chem.rename(columns={'name':'chem'})
    print(chem.head())

    for i in nega['id'].unique():
        df = pd.concat([posi[posi['id'] == 0], nega[nega['id'] == i]])
        df['id'] = (df['id'] == 0).astype(int)
        df = pd.merge(df, chem, on='chem')
        df = pd.merge(df, prot, on='prot')
        print(df)

if __name__ == '__main__':
    main()
