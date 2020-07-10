import numpy as np
import pandas as pd
import argparse
import gzip
import pybel
from rdkit import Chem
from rdkit.Chem import AllChem

def ecfp(args):
    with gzip.open('data/hiv.txt.gz', 'rt') as infile:
        lines = infile.readlines()

    fps = []
    for index, line in enumerate(lines, 1):
        smi, result = line.strip('\n').split(' ')
        mol = Chem.MolFromSmiles(smi)  

        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, nBits=args.nbits))
        fps.append(fp)

        print('\r%5d/%5d' % (index, len(lines)), end='')
    print('')
    
    df = pd.DataFrame(fps)
    print(df)
    df.to_csv('hiv.csv.gz', index=False)

def main(args):
    print('fptype = %s' % args.fptype)

    with gzip.open('data/hiv.txt.gz', 'rt') as infile:
        lines = infile.readlines()

    fps = []
    target = []
    for index, line in enumerate(lines, 1):
        try:
            mol = pybel.readstring('smi', line.strip('\n'))
        except Exception as e:
            continue
        fp = mol.calcfp(fptype=args.fptype).bits
        fps.append(fp)
        target.append(int(mol.title))

        print('\r%5d/%5d' % (index, len(lines)), end='')
    print('')

    ncol = max(max(fp) if len(fp) > 0 else 0 for fp in fps)
    mat = np.zeros((len(fps), ncol+1), dtype=np.int8)

    for i, fp in enumerate(fps):
        mat[i, fp] = 1

    columns = map(lambda x: 'f%04d' % x, range(ncol+1))

    df = pd.DataFrame(mat, columns=columns)
    df['target'] = target

    df = df.loc[:, df.sum() > 0]
    df = df.drop_duplicates()

    print(df)
    df.to_csv('hiv.csv.gz', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--fptype', default='ecfp', choices=['maccs', 'fp2', 'fp3', 'fp4', 'ecfp'])
    parser.add_argument('--radius', default=4, type=int)
    parser.add_argument('--nbits', default=1024, type=int)
    args = parser.parse_args()

    print(vars(args))

    if args.fptype == 'ecfp':
        ecfp(args)
    else:
        main(args)
