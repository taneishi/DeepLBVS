import pandas as pd
import numpy as np
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import sys

sys.path.append('../../dlvs')
import dnn

def build(diameter=4, nbits=2048):
    df = pd.read_csv('../../deepchem/datasets/pcba.csv.gz', sep=',')
    df = df.set_index(['mol_id','smiles'])
    df = df.to_sparse()
    df.to_pickle('data/pcba')

    mat = []
    for index,row in df.iterrows():
        mol_id, smiles = index
        mol = Chem.MolFromSmiles(smiles)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, int(diameter/2), nBits=nbits, useChirality=False,
            useBondTypes=True, useFeatures=False)
        fp = np.asarray(fp).astype(np.float32)
        fp[fp == 0] = np.nan
        mat.append(fp)

    df = pd.SparseDataFrame(mat, index=df.index)

    columns = range(nbits)
    columns = dict(zip(columns, map(lambda x: '%d_%d' % (diameter, x), columns)))
    df = df.rename(columns=columns)

    #df = df.to_sparse()
    df.to_pickle('data/ecfp%d_%d' % (diameter, nbits))
    sys.exit()

def main():
    build()
    np.random.seed(123)

    dnn.setup()
    dnn.show_version()

    diameter = 4
    nbits = 2048

    ecfp = pd.read_pickle('data/ecfp%d_%d' % (diameter, nbits))
    pcba = pd.read_pickle('data/pcba')

    for aid in pcba.columns:
        if aid == 'smiles': continue
        print('%s task start' % aid)

        cond = pcba[aid].notnull()
        df = pd.merge(ecfp, pcba.ix[cond, aid].to_frame(), left_index=True, right_index=True, how='inner')

        class_weight = (float(df.shape[0]) / df.groupby(df.ix[:,-1]).size()).to_dict()

        data = np.random.permutation(df.values)
        dnn.validation(aid, data, layers=[2000, 100],
                batch_size=1000, nb_epoch=100, class_weight=class_weight,
                optimizer='Adam', lr=0.0001, activation='relu', 
                dropout=0, patience=0, count=1)

if __name__ == '__main__':
    main()
