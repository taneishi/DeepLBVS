Prediction of Compound-Protein Interactions
===========================================

Environment
-----------

- Python 3.9
    - Anaconda3-2020.11
- PyTorch 1.8.0
    - conda install -c conda-forge pytorch
- RDKit
    - conda install -c conda-forge rdkit

Usage
-----

```
$ python main.py
```

Hyperparameters
---------------

```
$ python main.py -h

usage: main.py [-h] [--datafile DATAFILE] [--root_dir ROOT_DIR] [--random_seed RANDOM_SEED] [--test_size TEST_SIZE] [--model_dir MODEL_DIR] [--modelfile MODELFILE] [--epochs EPOCHS]
               [--batch_size BATCH_SIZE] [--lr LR] [--weight_decay WEIGHT_DECAY] [--dropout DROPOUT] [--cpu]

optional arguments:
  -h, --help            show this help message and exit
  --datafile DATAFILE
  --root_dir ROOT_DIR
  --random_seed RANDOM_SEED
  --test_size TEST_SIZE
  --model_dir MODEL_DIR
  --modelfile MODELFILE
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR
  --weight_decay WEIGHT_DECAY
  --dropout DROPOUT
  --cpu
```
