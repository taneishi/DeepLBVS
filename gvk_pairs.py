import pandas as pd
import numpy as np
import sqlite3

def main():
    db = sqlite3.connect('gvk/std_gpcr.db')
    tables = pd.read_sql_query("select name from sqlite_master where type = 'table';", db)
    print(tables)

    posi = pd.read_sql_query('select * from posi', db)
    
    print('Unordered positive pairs')
    print(posi[posi['id'] == 0])
    
    #print('Ordered positive pairs')
    #print(posi[posi['id'] == 1])

    nega = pd.read_sql_query('select * from nega', db)

    for i in nega['id'].unique():
        print('Negative pairs %d/5' % i)
        print(nega[nega['id'] == i])

    prot = pd.read_sql_query('select * from ptable', db)
    prot = prot.rename(columns={'name':'prot'})
    print(prot)

    chem = pd.read_sql_query('select * from ctable', db)
    chem = chem.rename(columns={'name':'chem'})
    print(chem)

    for i in nega['id'].unique():
        df = pd.concat([posi[posi['id'] == 0], nega[nega['id'] == i]])
        df = pd.merge(df, chem, on='chem')
        df = pd.merge(df, prot, on='prot')
        df = df.set_index(['id', 'chem', 'prot'])
        print(df)

if __name__ == '__main__':
    main()
