import pandas as pd
import numpy as np
from collections import defaultdict
import argparse

chemical = defaultdict(lambda: len(chemical))
gene = defaultdict(lambda: len(gene))

def main(args):
    filename = 'chemical_disease.tsv.gz'
    df = pd.read_csv('data/%s' % filename, sep='\t')

    df = df[df['DirectEvidence'].str.contains(args.evidence)]
    chemical_disease = df[df['DiseaseName'].str.contains(args.disease, case=False)]

    filename = 'gene_disease.tsv.gz'
    df = pd.read_csv('data/%s' % filename, sep='\t')

    df = df[df['DirectEvidence'].str.contains(args.evidence)]
    gene_disease = df[df['DiseaseName'].str.contains(args.disease, case=False)]

    filename = 'chemical_gene.tsv.gz'
    df = pd.read_csv('data/%s' % filename, sep='\t')

    df = df[df['ChemicalName'].isin(chemical_disease['ChemicalName']) &
            df['GeneSymbol'].isin(gene_disease['GeneSymbol'])]

    print(df)

    for i, row in df.iterrows():
        chemical[row['ChemicalName']]
        gene[row['GeneSymbol']]

    mat = np.zeros((len(chemical), len(gene)), dtype=np.int8)

    for index, (i, row) in enumerate(df.iterrows(), 1):
        print('\r%6d/%6d' % (index, df.shape[0]), end='')
        if pd.notnull(row['ChemicalName']) and pd.notnull(row['GeneSymbol']):
            mat[chemical[row['ChemicalName']], gene[row['GeneSymbol']]] = 1
    print('')

    df = pd.DataFrame(mat, index=chemical.keys(), columns=gene.keys())
    print(df)

    df.to_csv('data/mat.tsv', sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('disease')
    parser.add_argument('--evidence', choices=['therapuric', 'marker/mechanism', 'marker/mechanism|therapeutic'], default='therapeutic')
    args = parser.parse_args()

    main(args)
