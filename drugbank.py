import pandas as pd
import pybel

def main():
    drugbank = []
    for mol in pybel.readfile('sdf', 'data/drugbank.sdf'):
        drugbank_id = mol.data['DRUGBANK_ID']
        drugbank.append([drugbank_id, mol.write('smi').split('\t')[0]])

    drugbank = pd.DataFrame(drugbank)
    drugbank.columns = ['drugbank_id', 'smi']
    drugbank.to_csv('data/drugbank.tsv', sep='\t', index=False)
    print(drugbank)

if __name__ == '__main__':
    main()
