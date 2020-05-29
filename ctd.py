import pandas as pd
import pybel
import timeit

names = {
    'CTD_chem_gene_ixns': ['ChemicalName', 'ChemicalID', 'CasRN', 'GeneSymbol', 'GeneID', 'GeneForms', 'Organism', 'OrganismID', 'Interaction', 'InteractionActions', 'PubMedIDs'],
    'CTD_chemicals': ['ChemicalName', 'ChemicalID', 'CasRN', 'Definition', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'DrugBankIDs'],
    'CTD_chemicals_diseases': ['ChemicalName', 'ChemicalID', 'CasRN', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'InferenceGeneSymbol', 'InferenceScore', 'OmimIDs', 'PubMedIDs'],
    'CTD_diseases': ['DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Definition', 'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 'SlimMappings'],
    'CTD_genes': ['GeneSymbol', 'GeneName', 'GeneID', 'AltGeneIDs', 'Synonyms', 'BioGRIDIDs', 'PharmGKBIDs', 'UniProtIDs'],
    'CTD_genes_diseases': ['GeneSymbol', 'GeneID', 'DiseaseName', 'DiseaseID', 'DirectEvidence', 'InferenceChemicalName', 'InferenceScore', 'OmimIDs', 'PubMedIDs'],
    }

def main():
    filename = 'CTD_chem_gene_ixns.tsv.gz'
    print(filename)
    start_time = timeit.default_timer()
    df = pd.read_csv('data/%s' % filename, sep='\t', comment='#', names=names[filename[:-7]],
            usecols=['ChemicalName', 'GeneSymbol', 'Organism', 'InteractionActions'])
    print('%8d => ' % df.shape[0], end='')

    df.to_csv('data/chemical_gene.tsv.gz', sep='\t', index=False)
    print(' %6d %5.2f sec' % (df.shape[0], (timeit.default_timer() - start_time)))

    filename = 'CTD_chemicals_diseases.tsv.gz'
    print(filename)
    start_time = timeit.default_timer()
    df = pd.read_csv('data/%s' % filename, sep='\t', comment='#', names=names[filename[:-7]],
            usecols=['ChemicalName', 'DiseaseName', 'DirectEvidence'])
    print('%8d => ' % df.shape[0], end='')

    df = df[df['DirectEvidence'].notnull()]

    df.to_csv('data/chemical_disease.tsv.gz', sep='\t', index=False)
    print(' %6d %5.2f sec' % (df.shape[0], (timeit.default_timer() - start_time)))

    filename = 'CTD_genes_diseases.tsv.gz'
    print(filename)
    start_time = timeit.default_timer()
    df = pd.read_csv('data/%s' % filename, sep='\t', comment='#', names=names[filename[:-7]],
            usecols=['GeneSymbol', 'DiseaseName', 'DirectEvidence'], dtype={'DirectEvidence':str})
    print('%8d => ' % df.shape[0], end='')

    df = df[df['DirectEvidence'].notnull()]

    df.to_csv('data/gene_disease.tsv.gz', sep='\t', index=False)
    print(' %6d %5.2f sec' % (df.shape[0], (timeit.default_timer() - start_time)))

if __name__ == '__main__':
    main()
