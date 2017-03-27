Deep Learning for Virtual Screening
===================================

Dependency
----------

- deepchem-0.0.5.dev1449
    * https://github.com/deepchem/deepchem

- Tensorflow 1.0.1

- Anaconda 4.2.0

- Python 3.5.2

Usage
-----

    $ python chembl/tf_models.py
    ...
    Train scores
    {'mean-pearson_r2_score': 0.27899640394475578}
    Validation scores
    {'mean-pearson_r2_score': 0.3179097221346463}
    Test scores
    {'mean-pearson_r2_score': 0.34468870141844038}
    Train time: 17.1m
    Eval time: 1.0m

    $ python chembl/graph_conv.py
    ...
    Train scores
    {'mean-pearson_r2_score': 0.14758438301407914}
    Validation scores
    {'mean-pearson_r2_score': 0.21988992589239506}
    Test scores
    {'mean-pearson_r2_score': 0.21277860356454578}
    Train time: 17.7m
    Eval time: 1.4m

    # python pcba/tf_models.py
    ...
    Train scores
    {'mean-roc_auc_score': 0.93896412511755167}
    Validation scores
    {'mean-roc_auc_score': 0.79112713308332894}
    Test scores
    {'mean-roc_auc_score': 0.7915215860043836}
    Train time: 87.8m
    Eval time: 14.5m

    # python pcba/graph_conv.py
    Train scores
    {'mean-roc_auc_score': 0.9175177701552093}
    Validation scores
    {'mean-roc_auc_score': 0.86004011415477311}
    Test scores
    {'mean-roc_auc_score': 0.85849436864485928}
    Train time: 217.0m
    Eval time: 32.1m
