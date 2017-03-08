Deep Learning for Virtual Screening
===================================

Dependency
----------

- deepchem-0.0.5.dev1357
    * https://github.com/deepchem/deepchem

- Tensorflow 0.12.1

- Anaconda 4.3.0

- Python 3.5.2

Usage
-----

    $ python chembl/tf_models.py
    ...
    Train scores
    {'mean-pearson_r2_score': 0.2800800558893527}
    Validation scores
    {'mean-pearson_r2_score': 0.31851526805470631}
    Test scores
    {'mean-pearson_r2_score': 0.34494307106309441}
    Train time: 25.3m
    Eval time: 1.0m

    $ python chembl/graph_conv.py
    ...
    Train scores
    {'mean-pearson_r2_score': 0.14157082738050894}
    Validation scores
    {'mean-pearson_r2_score': 0.22814428897445027}
    Test scores
    {'mean-pearson_r2_score': 0.2826175439387309}
    Train time: 19.6m
    Eval time: 1.4m

    # python pcba/tf_models.py
    ...
    Train scores
    {'mean-roc_auc_score': 0.9433418080019409}
    Validation scores
    {'mean-roc_auc_score': 0.79813118519108261}
    Test scores
    {'mean-roc_auc_score': 0.79524128220494039}
    Train time: 124.0m
    Eval time: 21.3m
