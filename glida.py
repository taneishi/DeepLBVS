import pandas as pd
import os

def main():
    input_dir = '../data/glida'
    output_dir = 'data'

    gpcr = pd.read_csv(os.path.join(input_dir, 'gpcr.tsv.gz'), sep='\t')
    gpcr_fasta = pd.read_csv(os.path.join(input_dir, 'gpcr_fasta.tsv.gz'), sep='\t')

    gpcr = pd.merge(gpcr, gpcr_fasta, on='name', how='left')
    print(gpcr)

    ligand = pd.read_csv(os.path.join(input_dir, 'ligand.tsv.gz'), sep='\t')
    ligand_smi = pd.read_csv(os.path.join(input_dir, 'ligand_smi.tsv.gz'), sep='\t')

    ligand = pd.merge(ligand, ligand_smi, on='name', how='left')
    print(ligand)

    relation = pd.read_csv(os.path.join(input_dir, 'relation.tsv.gz'), sep='\t')

    relation = pd.merge(relation[['ligand_id', 'gpcr_id', 'activity']], gpcr[['gpcr_id', 'fasta']], on='gpcr_id', how='left')
    relation = pd.merge(relation, ligand[['ligand_id', 'smi']], on='ligand_id', how='left')

    print(relation)
    relation.to_csv(os.path.join(output_dir, 'gpcr-ligand.tsv.gz'), index=False, sep='\t')

if __name__ == '__main__':
    main()
