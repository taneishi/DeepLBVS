import theano
import theano.tensor as T
import numpy as np
import time

def gpu():
    vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
    iters = 1000

    rng = np.random.RandomState(22)
    x = theano.shared(np.asarray(rng.rand(vlen), theano.config.floatX))
    f = theano.function([], T.exp(x))

    t0 = time.time()
    for i in xrange(iters):
        r = f()
    t1 = time.time()
    print 'Looping %d times took' % iters, t1 - t0, 'seconds'
    print 'Result is', r
    if np.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
        print 'Used the cpu'
    else:
        print 'Used the gpu'

def power():
    np.set_printoptions(precision=2,suppress=True,linewidth=120)

    k = T.iscalar("k")
    A = T.vector("A")

    # Symbolic description of the result
    result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                                  outputs_info=T.ones_like(A),
                                  #outputs_info=A, #T.ones_like(A),
                                  non_sequences=A,
                                  n_steps=k)

    # We only care about A**k, but scan has provided us with A**1 through A**k.
    # Discard the values that we don't care about. Scan is smart enough to
    # notice this and not waste memory saving them.
    #final_result = result[-1]
    final_result = result

    # compiled function that returns A**k
    power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)

    a = power(np.arange(10,dtype=theano.config.floatX),5)
    print a

if __name__ == '__main__':
    gpu()
    power()
