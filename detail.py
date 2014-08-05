import cPickle
import gzip
import numpy
import sys

def detail(data,name):
    nrow,ncol = data[0].shape
    print '%s (%dx%d) min:%.2f max:%.2f' % (name,nrow,ncol,numpy.min(data[0]),numpy.max(data[0]))
    for i in sorted(set(data[1])):
        print '%s:%d,' % (i, len(filter(lambda x:x==i, data[1]))),
    print

def main():
    if len(sys.argv) == 1: sys.exit('%s [filename]' % sys.argv[0])
    fn = sys.argv[1]
    train,valid,test = cPickle.load(gzip.open(fn))

    detail(train,'train')
    detail(valid,'valid')
    detail(test,'test')

main()
