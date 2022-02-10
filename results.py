import pandas as pd
import numpy as np
import argparse
import os

def main(args):
    results = dict()

    diameter_list, nbits_list = [], []
    for filename in os.listdir(args.log_dir):
        diameter, nbits, basename = filename.split('_')
        diameter_list.append(int(diameter))
        nbits_list.append(int(nbits))

    for diameter in sorted(diameter_list):
        for nbits in sorted(nbits_list):
            filename = '%s/%d_%d_results.tsv.gz' % (args.log_dir, diameter, nbits)
            if os.path.exists(filename):
                df = pd.read_csv(filename, sep='\t', index_col=0)
                results.update({'%d_%d' % (diameter, nbits): df['MeanAUC']})
        
    df = pd.DataFrame.from_dict(results)
    df.loc['mean', :] = df.mean(axis=0)
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='log', type=str)
    args = parser.parse_args()
    print(vars(args))

    main(args)
