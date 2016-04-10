# coding:utf-8
import pandas as pd
import numpy as np
from sklearn.feature_extraction import FeatureHasher
import os

DIR = 'gdahl'

def identifiers_build():
    identifiers = set()
    if os.path.isfile('identifiers'):
        for l in open('identifiers'):
            identifiers.add(int(l.strip()))
    for filename in sorted(os.listdir(DIR)):
        for l in open(os.path.join(DIR, '%s/ECFP/ECFP.list' % filename)):
            seq = l.strip().split()
            seq = map(lambda x: int(x), seq[1:])
            identifiers.update(seq)
        print filename, len(identifiers)
    out = open('identifiers', 'a')
    for i in identifiers:
        out.write('%d\n' % i)
    out.close()

def identifiers_mapper():
    if not os.path.exists('identifiers'):
        identifiers_build()
    identifiers = set()
    for l in open('identifiers'):
        identifiers.add(int(l.strip()))
    return dict(zip(identifiers, range(len(identifiers))))

def feature_mapper(outdir='ecfp', name='ECFP'):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    mapper = identifiers_mapper()
    print len(mapper)

    for filename in sorted(os.listdir(DIR)):
        activity = pd.read_csv(os.path.join(DIR, '%s/assay.csv' % filename), sep='\t', header=None)
        activity[1] = (activity[1] == 'active').astype(int)
        activity = activity.set_index(0)[1].to_dict()
        mat = []
        for l in open(os.path.join(DIR, '%s/%s/%s.list' % (filename, name, name))):
            arr = np.zeros(len(mapper)+1, dtype=np.int8)
            seq = l.strip().split()
            arr[-1] = activity[int(seq[0])]
            for x in seq[1:]:
                arr[mapper[int(x)]] = 1
            mat.append([arr])
        mat = np.concatenate(mat)
        df = pd.DataFrame(mat, dtype=np.int8)
        df.to_pickle(os.path.join(outdir, filename))
        print filename, df.shape

def feature_hasher(n_features=3000):
    if not os.path.isdir('ecfp%d' % n_features):
        os.makedirs('ecfp%d' % n_features)

    hasher = FeatureHasher(n_features=n_features, non_negative=True)
    for filename in sorted(os.listdir(DIR)):
        if os.path.isfile('ecfp%d/%s' % (n_features, filename)):
            continue
        activity = pd.read_csv(os.path.join(DIR, '%s/assay.csv' % filename), sep='\t', header=None)
        activity[1] = (activity[1] == 'active').astype(int)
        activity = activity.set_index(0)[1].to_dict()
        mat = []
        target = []
        for l in open(os.path.join(DIR, '%s/ECFP/ECFP.list' % filename)):
            seq = l.strip().split()
            target.append(activity[int(seq[0])])
            seq = map(lambda x: (x,1), seq[1:])
            seq = hasher.transform([dict(seq)])
            mat.append(seq.toarray())
        mat = np.concatenate(mat)
        df = pd.DataFrame(mat, dtype=np.int8)
        df['target'] = target
        df.to_pickle('ecfp%d/%s' % (n_features, filename))
        print filename, df.shape

def dragon_mapper(outdir='dragon', name='Dragon_2D'):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    for filename in sorted(os.listdir(DIR)):
        activity = pd.read_csv(os.path.join(DIR, '%s/assay.csv' % filename), sep='\t', header=None)
        activity[1] = (activity[1] == 'active').astype(int)
        activity = activity.set_index(0)[1].to_dict()

        df = pd.read_csv(os.path.join(DIR, '%s/%s/%s.list' % (filename, name, name)), sep='\t', header=None, index_col=0, dtype=np.float32)
        df['target'] = df.index.map(lambda x: activity.get(x, None))

        df.to_pickle(os.path.join(outdir, filename))
        print filename, df.shape


if __name__ == '__main__':
    #dragon_mapper()
    #feature_mapper()
    feature_hasher()
