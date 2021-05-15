import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse
import gzip

def main(args):
    table = []
    fps = []
    bit_info = {}
    molecules = list(Chem.ForwardSDMolSupplier(gzip.open('%s/%s' % (args.input_dir, args.filename))))
    for index, mol in enumerate(molecules, 1):
        if not mol:
            continue

        drugbank_id = mol.GetProp('DRUGBANK_ID')
        smi = Chem.MolToSmiles(mol)
        table.append([drugbank_id, smi])
    
        fp_bit = list(AllChem.GetMorganFingerprintAsBitVect(mol, args.radius, bitInfo=bit_info, nBits=args.nbits))

        # you can use fp_bit for fixed length fingerprints,
        # while this code generates matrix from variable length fingerprints for generality.
        fp = list(bit_info.keys())
        fps.append(fp)

        print('\r%5d/%5d' % (index, len(molecules)), end='')

    print('\n')

    # for variable length
    ncol = max(max(fp) if len(fp) > 0 else 0 for fp in fps)
    mat = np.zeros((len(fps), ncol+1), dtype=np.int8)

    for i, fp in enumerate(fps):
        mat[i, fp] = 1
    
    df = pd.concat([pd.DataFrame(table, columns=['drugbank_id', 'smi']), pd.DataFrame(mat)], axis=1)
    print(df)
    df.to_csv('%s/drugbank.tsv.gz' % (args.output_dir), index=False, sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--filename', default='drugbank.sdf.gz')
    #filename = 'data/hiv.txt'
    parser.add_argument('--input_dir', default='../data/drugbank')
    parser.add_argument('--output_dir', default='data')
    parser.add_argument('--radius', default=2, type=int)
    parser.add_argument('--nbits', default=1024, type=int)
    args = parser.parse_args()
    print(vars(args))

    main(args)
