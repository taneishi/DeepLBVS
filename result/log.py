# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) < 2:
        sys.exit('%s [filename]' % sys.argv[0])

    plt.figure(figsize=(8,6),dpi=100)

    for filename in sys.argv[1:]:
        print filename
        df = pd.read_pickle(filename)
        df['epoch'] = df.index.astype(int)
        df = df.sort('epoch')
        df = df.set_index('epoch')


        title = filename.replace('GPCR_1st','').replace('_',' ')[:-4] + ' interactions'
        #plt.title(title)

        valid = df['valid'].dropna() * 100.
        plt.plot(valid.index, valid, label='validation / ' + title) 
        test = df['test'].dropna() * 100.
        plt.plot(test.index, test, label='test / ' + title, linewidth=2) 

    plt.ylim(0,50)
    plt.ylabel('Error rate (%)')
    plt.xlim(0,500)
    plt.xlabel('Epochs (iterations)')
    plt.title('GPCR')
    plt.legend()
    plt.tight_layout()

    #plt.savefig(filename.replace('log','png'))
    #plt.savefig('gpcr.png')
    plt.show()

if __name__ == '__main__':
    main()
