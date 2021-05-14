import pandas as pd
from rdkit import Chem
import gzip

def main():
    input_dir = '../data/drugbank'
    output_dir = 'data'

    drugbank = []
    for mol in Chem.ForwardSDMolSupplier(gzip.open('%s/drugbank.sdf.gz' % (input_dir))):
        if mol:
            drugbank_id = mol.GetProp('DRUGBANK_ID')
            drugbank.append([drugbank_id, Chem.MolToSmiles(mol)])

    drugbank = pd.DataFrame(drugbank)
    drugbank.columns = ['drugbank_id', 'smi']
    drugbank.to_csv('%s/drugbank.tsv' % (output_dir), sep='\t', index=False)
    print(drugbank)

if __name__ == '__main__':
    main()
