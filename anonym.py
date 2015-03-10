
import pandas as pd
import os

def main():
    for i in xrange(1,6):
        filename = os.path.join('data','out%d' % i)
        df = pd.read_pickle(filename)

        df = df.reset_index()
        del df[0]
        print df.head()
        df.to_pickle(os.path.join(os.environ['HOME'], 'data', 'dataset%d.pkl' % i))


if __name__ == '__main__':
    main()
