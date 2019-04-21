Deep Learning for Virtual Screening
===================================

Dependency
----------

- deepchem 0.0.5.dev2704
    * https://github.com/deepchem/deepchem

- Tensorflow v1.3.0-rc2-20-g0787eee

- Anaconda 4.3

- Python 3.6.3


Method
------

- Multi-task DNN and Graph convolutional network

- 5-fold CV


Usage
-----

    $ make
    
    ...
     
    Tox21 dataset
    Multi-task DNN
    Train scores: 0.848936655301
    Test scores: 0.789951449973
    Train time: 2.2m
    Graph-Convolution
    Train scores: 0.90177625967
    Test scores: 0.825843831165
    Train time: 14.6m

![tox21 plot](https://raw.githubusercontent.com/ktaneishi/dlvs/master/log/tox21.png)
