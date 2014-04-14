import numpy
import gzip
import cPickle
import random
import os

def scale(mat):
    for i in xrange(mat.shape[1]):
        v = mat[:,i]
        if numpy.max(v)-numpy.min(v) != 0:
            mat[:,i] = (v - numpy.min(v)) / (numpy.max(v)-numpy.min(v))
        else:
            mat[:,i] = v * 0
    return mat

def matrix(fn):
    labels = []
    matrix = []
    print fn
    for l in open(fn):
        seq = l.strip().split(' ')
        labels.append(seq.pop(0))
        seq = map(float,seq)
        matrix.append(seq)

    mat = numpy.matrix(matrix)  
    mat = scale(mat)
    mat = numpy.array(mat,dtype=numpy.float32)

    cls = {}
    for l in open(fn.replace('cgbvs-', 'cpis_').replace('trial1_', '').replace('vec', 'trial_1').replace('5000', '_5000')):
        l = l.strip('()\n')
        l = l.replace('chemID=','').replace('protID=','').replace('status=','').replace('non-interaction','0').replace('interaction','1')
        seq = l.split(', ')
        label = '%s--%s' % (seq[0],seq[1])
        cls[label] = int(seq[2])
        
    cls = numpy.array(map(lambda x: cls[x], labels),dtype=numpy.int64)
    return (mat,cls)

def pklgz(DIR):
    files = sorted(os.listdir(DIR),reverse=True)
    files = filter(lambda x:x.endswith('.vec.txt'), files)
    train,test = map(lambda x:os.path.join(DIR,x), files)
    
    fn = train.replace('cgbvs-','').replace('_trial1_train.vec.txt','')+'.gz'

    train = matrix(train)
    test = matrix(test)

    nrow,ncol = train[0].shape
    sample = random.sample(xrange(nrow), nrow/6)
    valid = (train[0][sample,],train[1][sample])

    sample = list(set(range(nrow)) - set(sample))
    train = (train[0][sample,],train[1][sample]) 

    out = gzip.open(DIR+fn, 'wb')
    cPickle.dump((train,valid,test), out)
    out.close()

def main():
    dirs = ['gpcr', 'kinase']
    for DIR in dirs:
        pklgz(DIR)

main()
