Deep Learning for Virtual Screening
===================================

Dependency
----------

- deepchem 0.0.5.dev2704
    * https://github.com/deepchem/deepchem

- Tensorflow v1.3.0-rc2-20-g0787eee

- Anaconda 4.3

- Python 3.6.3

Usage
-----

    $ make
    
    ...
     
    PCBA dataset
    Multi-task DNN
    Train scores: 0.819380206847
    Test scores: 0.789931466539
    Train time: 608.6m
    Graph-Convolution
    Train scores: 0.889408119487
    Test scores: 0.850056521039
    Train time: 1958.0m

![pcba plot](https://raw.githubusercontent.com/ktaneishi/dlvs/master/log/pcba.png)

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

    Delaney dataset
    Multi-task DNN
    Train scores: {'mean-pearson_r2_score': 0.86165358015042837}
    Validation scores: {'mean-pearson_r2_score': 0.61665541324854578}
    Train time: 0.1m
    Eval time: 0.0m
    Graph-Convolution
    Train scores: {'mean-pearson_r2_score': 0.97126488150811363}
    Validation scores: {'mean-pearson_r2_score': 0.83998886621722046}
    Train time: 0.2m
    Eval time: 0.0m

![delaney plot](https://raw.githubusercontent.com/ktaneishi/dlvs/master/log/delaney.png)

