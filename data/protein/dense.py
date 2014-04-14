import gzip
import cPickle
import numpy

def dense(filename,ncol):
    labels = []
    matrix = []
    for l in open(filename):
        seq = l.strip().replace(': ',':').split(' ')
        seq = filter(lambda x: x != '', seq)
        labels.append(int(seq.pop(0)))
        v = numpy.zeros(ncol, dtype=numpy.float32)
        for col,val in map(lambda x:x.split(':'), seq):
            if int(col)-1 < ncol:
                v[int(col)-1] = float(val)
        matrix.append(v.tolist())
    mat = numpy.array(matrix, dtype=numpy.float32)
    cls = numpy.array(labels, dtype=numpy.int64)
    return (mat,cls)

def main():
    ncol = 358
    train = dense('protein.tr',ncol)
    valid = dense('protein.val',ncol)
    test = dense('protein.t',ncol)

    out = gzip.open('protein.gz','wb')
    cPickle.dump((train,valid,test), out)
    out.close()

main()
