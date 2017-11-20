Deep Learning for Virtual Screening
===================================

Dependency
----------

- deepchem-0.0.5.dev1902
    * https://github.com/deepchem/deepchem

- Tensorflow 1.0.1

- Anaconda 4.3-base

- Python 3.6.0

Usage
-----

    $ make
    
    ...
     

    PCBA dataset
    Multi-task DNN
    Train scores: {'mean-roc_auc_score': 0.81701982308592269}
    Validation scores: {'mean-roc_auc_score': 0.7923668677318576}
    Test scores: {'mean-roc_auc_score': 0.79004576329170462}
    Train time: 36.9m
    Eval time: 5.6m
    Graph-Convolution
    Train scores: {'mean-roc_auc_score': 0.88851156241465601}
    Validation scores: {'mean-roc_auc_score': 0.85417880493997422}
    Test scores: {'mean-roc_auc_score': 0.85218386002296209}
    Train time: 105.2m
    Eval time: 10.2m

![pcba plot](https://raw.githubusercontent.com/ktaneishi/dlvs/master/log/pcba.png)

    Tox21 dataset
    Multi-task DNN
    Train scores: {'mean-roc_auc_score': 0.85107537796311739}
    Validation scores: {'mean-roc_auc_score': 0.7751755026286723}
    Train time: 0.2m
    Eval time: 0.0m
    Graph-Convolution
    Train scores: {'mean-roc_auc_score': 0.89637726248386951}
    Validation scores: {'mean-roc_auc_score': 0.82233404602065086}
    Train time: 1.5m
    Eval time: 0.2m

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

