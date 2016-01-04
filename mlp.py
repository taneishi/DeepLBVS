# Multilayer perceptron

import os
import sys
import time

import numpy

import theano
import theano.tensor as T

from utils import load_data

from code.logistic_sgd import LogisticRegression
from code.mlp import HiddenLayer
import pandas as pd
import numpy as np

from sklearn.metrics import roc_curve, auc

class MLP(object):
    def __init__(self, rng, input, n_in, hidden_layers_sizes, n_out):
        self.hidden_layers = []
        self.n_layers = len(hidden_layers_sizes)
        self.params = []
        self.L1 = 0
        self.L2_sqr = 0

        assert self.n_layers > 0

        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layers_sizes[i-1]

            if i == 0:
                layer_input = input
            else:
                layer_input = self.hidden_layers[-1].output

            hidden_layer = HiddenLayer(
                rng=rng,
                input=layer_input,
                n_in=input_size,
                n_out=hidden_layers_sizes[i],
                activation=T.nnet.sigmoid
            )

            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)
            self.L1 = (self.L1 + abs(hidden_layer.W).sum())
            self.L2_sqr = (self.L2_sqr + (hidden_layer.W ** 2).sum())

        self.logRegressionLayer = LogisticRegression(
            input=self.hidden_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_out
        )

        self.L1 = (self.L1 + abs(self.logRegressionLayer.W).sum())

        self.L2_sqr = (self.L2_sqr + (self.logRegressionLayer.W ** 2).sum())

        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )

        self.errors = self.logRegressionLayer.errors

        self.params.extend(self.logRegressionLayer.params)

        self.input = input

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, hidden_layers_sizes=[1000]):

    basename = 'mlp/%s_%s_%d_%f.log' % (
            os.path.basename(dataset),
            '_'.join(map(str, hidden_layers_sizes)),
            batch_size, learning_rate)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    n_in = train_set_x.shape[1].eval()
    n_out = len(set(train_set_y.eval()))

    # compute number of minibatches for training and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(123)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=n_in,
        hidden_layers_sizes=hidden_layers_sizes,
        n_out=n_out
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.logRegressionLayer.y_pred)

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    test_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the test set; in this case we
                                  # check every epoch

    best_test_score = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    score = []

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % test_frequency == 0:
                # compute zero-one loss on test set
                test_losses = [test_model(i) for i
                                     in xrange(n_test_batches)]
                this_test_score = numpy.mean(test_losses)
                score.append([epoch,this_test_score])

                #res['STATUS'] = numpy.concatenate([mat[2] for mat in test_result])
                #fpr,tpr,thresholds = roc_curve(res['STATUS'], res[1], pos_label=1)
                #print auc(fpr, tpr)
                predicted_values = predict_model(test_set_x.get_value())
                #print test_set_y.eval()
                #print predicted_values

                print(
                    'epoch %i, minibatch %i/%i, test error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_test_score * 100.
                    )
                )

                # if we got the best test score until now
                if this_test_score < best_test_score:
                    #improve patience if loss improvement is good enough
                    if (
                        this_test_score < best_test_score *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_test_score = this_test_score
                    best_iter = iter

            if patience <= iter:
                #done_looping = True
                #break
                pass

    # save weights
    params = {}
    for i in xrange(0, len(classifier.params)/2):
        params['W%d' % i] = classifier.params[i*2].get_value()
        params['b%d' % i] = classifier.params[i*2+1].get_value()
    np.savez(os.path.join('model', basename).replace('log','npz'), params)

    df = pd.DataFrame(score)
    df.to_pickle(os.path.join('result', basename))

    end_time = time.clock()
    print(('Optimization complete. Best test performance %f %% obtained at iteration %i') %
          (best_test_score * 100., best_iter + 1))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        sys.exit('Usage: %s [datafile]' % (sys.argv[0]))

    test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.00, n_epochs=1000,
            dataset=dataset, batch_size=100, hidden_layers_sizes=[1000,1000,1000])
