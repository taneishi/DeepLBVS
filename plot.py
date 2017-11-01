import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set()

def main():
    for dataset in ['chembl', 'pcba', 'tox21', 'delaney']:
        plt.figure(figsize=(8,4))
        for i, method in enumerate(['tf_models', 'graph_conv'], 1):  
            filename = os.path.join('log', dataset, method) + '.pkl'
            if not os.path.exists(filename):
                continue
            df = pd.read_pickle(filename)

            ax = plt.subplot(1,2,i)
            df.boxplot(ax=ax)
            if dataset == 'chembl':
                if i == 1:
                    plt.ylabel('Pearson R-squared')
                plt.ylim(0.,1.)
            elif dataset in ['pcba', 'tox21', 'delaney']:
                if i == 1:
                    plt.ylabel('ROC AUC')
                plt.ylim(0.5, 1.0)
            plt.title('%s %s' % (dataset, method))
            plt.tight_layout()
        plt.savefig('log/%s.png' % (dataset))

if __name__ == '__main__':
    main()
