import pandas as pd

def main():
    gpcr = pd.read_csv('data/gpcr.txt', sep='\t')
    gpcr_fasta = pd.read_csv('data/gpcr_fasta.txt', sep='\t')

    gpcr = pd.merge(gpcr, gpcr_fasta, on='name', how='left')
    print(gpcr)

    ligand = pd.read_csv('data/ligand.txt', sep='\t')
    ligand_smi = pd.read_csv('data/ligand_smi.txt', sep='\t')

    ligand = pd.merge(ligand, ligand_smi, on='name', how='left')
    print(ligand)

    relation = pd.read_csv('data/relation.txt', sep='\t')

    relation = pd.merge(relation[['ligand_id', 'gpcr_id', 'activity']], gpcr[['gpcr_id', 'fasta']], on='gpcr_id', how='left')
    relation = pd.merge(relation, ligand[['ligand_id', 'smi']], on='ligand_id', how='left')
    print(relation)

if __name__ == '__main__':
    main()
