import numpy as np
import pandas as pd
import argparse
import gzip
from rdkit import Chem
from rdkit.Chem import AllChem

def main(args):
    filename = 'data/drugbank.tsv'
    #filename = 'data/hiv.txt'

    df = pd.read_csv(filename, sep='\t')
    print(df)

    fps = []
    bit_info = {}
    for index, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smi'])

        if not mol:
            continue

        fp_bit = list(AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, bitInfo=bit_info, nBits=args.nbits))

        # you can use fp_bit for fixed length fingerprints,
        # while this code generates matrix from variable length fingerprints for generality.
        fp = list(bit_info.keys())
        fps.append(fp)

        print('\r%5d/%5d' % (index, df.shape[0]), end='')

    print('\nfinished')

    # for variable length
    ncol = max(max(fp) if len(fp) > 0 else 0 for fp in fps)
    mat = np.zeros((len(fps), ncol+1), dtype=np.int8)

    for i, fp in enumerate(fps):
        mat[i, fp] = 1
    
    df = pd.DataFrame(mat)
    df.to_csv('data/ecfp.csv', index=False)
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--radius', default=2, type=int)
    parser.add_argument('--nbits', default=1024, type=int)
    args = parser.parse_args()

    print(vars(args))

    main(args)
