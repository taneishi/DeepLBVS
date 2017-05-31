Deep Learning for Virtual Screening
===================================

Dependency
----------

- deepchem-0.0.5.dev1780
    * https://github.com/deepchem/deepchem

- Tensorflow 1.0.1

- Anaconda 4.3 based

- Python 3.6.0

Usage
-----

    $ make
    
    ...
     
    ChEMBL dataset
    Train scores: {'mean-pearson_r2_score': 0.8263672189829635}
    Validation scores: {'mean-pearson_r2_score': 0.56835987646868558}
    Test scores: {'mean-pearson_r2_score': 0.59474859997338803}
    Train time: 46.3m
    Eval time: 0.7m

    Train scores: {'mean-pearson_r2_score': 0.14826334786558712}
    Validation scores: {'mean-pearson_r2_score': 0.23758873756708912}
    Test scores: {'mean-pearson_r2_score': 0.26867242413579084}
    Train time: 13.2m
    Eval time: 0.6m

![chembl plot](https://raw.githubusercontent.com/ktaneishi/dlvs/master/log/chembl.png)
    
    PCBA dataset
    Train scores: {'mean-roc_auc_score': 0.80845857532109999}
    Validation scores: {'mean-roc_auc_score': 0.78916964022602043}
    Test scores: {'mean-roc_auc_score': 0.77831819933338153}
    Train time: 105.7m
    Eval time: 17.2m

    Train scores: {'mean-roc_auc_score': 0.88522551045202269}
    Validation scores: {'mean-roc_auc_score': 0.85121845799474638}
    Test scores: {'mean-roc_auc_score': 0.84671217941655996}
    Train time: 208.3m
    Eval time: 17.4m

![pcba plot](https://raw.githubusercontent.com/ktaneishi/dlvs/master/log/pcba.png)
