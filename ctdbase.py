import pandas as pd
import numpy as np
from collections import defaultdict
import argparse
import timeit
import os

names = {
    'CTD_chem_gene_ixns': ['ChemicalName', 'ChemicalID', 'CasRN', 'GeneSymbol', 'GeneID', 'GeneForms', 'Organism', 'OrganismID', 'Interaction', 'InteractionActions', 'PubMedIDs'],
    'CTD_chemicals': ['ChemicalName', 'ChemicalID', 'CasRN', 'Definition', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'DrugBankIDs'],
    'CTD_chemicals_diseases': ['ChemicalName', 'ChemicalID', 'CasRN', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'InferenceGeneSymbol', 'InferenceScore', 'OmimIDs', 'PubMedIDs'],
    'CTD_diseases': ['DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Definition', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'SlimMappings'],
    'CTD_genes': ['GeneSymbol', 'GeneName', 'GeneID', 'AltGeneIDs', 'Synonyms', 'BioGRIDIDs', 'PharmGKBIDs', 'UniProtIDs'],
    'CTD_genes_diseases': ['GeneSymbol', 'GeneID', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'InferenceChemicalName', 'InferenceScore', 'OmimIDs', 'PubMedIDs'],
    }

chemical = defaultdict(lambda: len(chemical))
gene = defaultdict(lambda: len(gene))

def main(args):
    filename = 'CTD_chem_gene_ixns.tsv.gz'
    print(filename)
    start_time = timeit.default_timer()
    df = pd.read_csv(os.path.join(args.input_dir, filename), sep='\t', comment='#', names=names[filename[:-7]],
            usecols=['ChemicalName', 'GeneSymbol', 'Organism', 'InteractionActions'])
    print('%8d => ' % df.shape[0], end='')

    df.to_csv(os.path.join(args.output_dir, 'chemical_gene.tsv.gz'), sep='\t', index=False)
    print(' %6d %5.2f sec' % (df.shape[0], (timeit.default_timer() - start_time)))

    filename = 'CTD_chemicals_diseases.tsv.gz'
    print(filename)
    start_time = timeit.default_timer()
    df = pd.read_csv(os.path.join(args.input_dir, filename), sep='\t', comment='#', names=names[filename[:-7]],
            usecols=['ChemicalName', 'DiseaseName', 'DirectEvidence'])
    print('%8d => ' % df.shape[0], end='')

    df = df[df['DirectEvidence'].notnull()]

    df.to_csv(os.path.join(args.output_dir, 'chemical_disease.tsv.gz'), sep='\t', index=False)
    print(' %6d %5.2f sec' % (df.shape[0], (timeit.default_timer() - start_time)))

    filename = 'CTD_genes_diseases.tsv.gz'
    print(filename)
    start_time = timeit.default_timer()
    df = pd.read_csv(os.path.join(args.input_dir, filename), sep='\t', comment='#', names=names[filename[:-7]],
            usecols=['GeneSymbol', 'DiseaseName', 'DirectEvidence'], dtype={'DirectEvidence':str})
    print('%8d => ' % df.shape[0], end='')

    df = df[df['DirectEvidence'].notnull()]

    df.to_csv(os.path.join(args.output_dir, 'gene_disease.tsv.gz'), sep='\t', index=False)
    print(' %6d %5.2f sec' % (df.shape[0], (timeit.default_timer() - start_time)))

def matrix(args):
    filename = 'chemical_disease.tsv.gz'
    df = pd.read_csv(os.path.join(args.output_dir, filename), sep='\t')

    df = df[df['DirectEvidence'].str.contains(args.evidence)]
    chemical_disease = df[df['DiseaseName'].str.contains(args.disease, case=False)]

    filename = 'gene_disease.tsv.gz'
    df = pd.read_csv(os.path.join(args.output_dir, filename), sep='\t')

    df = df[df['DirectEvidence'].str.contains(args.evidence)]
    gene_disease = df[df['DiseaseName'].str.contains(args.disease, case=False)]

    filename = 'chemical_gene.tsv.gz'
    df = pd.read_csv(os.path.join(args.output_dir, filename), sep='\t')

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

    df.to_csv(os.path.join(args.output_dir, 'mat.tsv.gz'), sep='\t')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='../data/ctdbase')
    parser.add_argument('--output_dir', default='data')
    parser.add_argument('--disease', default='diabetes')
    parser.add_argument('--evidence', choices=['therapuric', 'marker/mechanism', 'marker/mechanism|therapeutic'], default='therapeutic')
    args = parser.parse_args()
    print(vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
    matrix(args)

