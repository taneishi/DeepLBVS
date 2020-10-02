import pandas as pd
from rdkit import Chem

def main():
    drugbank = []
    for mol in Chem.SDMolSupplier('data/drugbank.sdf'):
        if mol:
            drugbank_id = mol.GetProp('DRUGBANK_ID')
            drugbank.append([drugbank_id, Chem.MolToSmiles(mol)])

    drugbank = pd.DataFrame(drugbank)
    drugbank.columns = ['drugbank_id', 'smi']
    drugbank.to_csv('data/drugbank.tsv', sep='\t', index=False)
    print(drugbank)

if __name__ == '__main__':
    main()
