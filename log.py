import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    plt.figure(figsize=(8,6),dpi=100)

    for filename in sys.argv[1:]:
        print(filename)
        df = pd.read_pickle(filename)
        df['epoch'] = df.index.astype(int)
        df = df.sort('epoch')
        df = df.set_index('epoch')

        label = filename.replace('GPCR_1st','').replace('_',' ')[:-4] + ' interactions'

        valid = df['valid'].dropna() * 100.
        plt.plot(valid.index, valid, label='validation / %s' % (label)) 
        test = df['test'].dropna() * 100.
        plt.plot(test.index, test, label='test / %s' % (label), linewidth=2) 

    plt.ylim(0,50)
    plt.ylabel('Error rate (%)')
    plt.xlim(0,500)
    plt.xlabel('Epochs (iterations)')
    plt.title('GPCR')
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
