Deep Learning for Multi-modal Virtual Screening
===============================================

Requirements
------------

- Python3.9
    - Anaconda3-2020.11
- PyTorch1.8.0
    - conda install -c conda-forge pytorch
- RDKit
    - conda install -c conda-forge rdkit

Usage
-----

```
$ python main.py -h

usage: main.py [-h] [--datafile DATAFILE] [--modelfile MODELFILE]
               [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
               [--weight_decay WEIGHT_DECAY] [--dropout DROPOUT]
               [--random_seed RANDOM_SEED] [--cpu]

optional arguments:
  -h, --help            show this help message and exit
  --datafile DATAFILE
  --modelfile MODELFILE
  --epochs EPOCHS
  --batch_size BATCH_SIZE
  --lr LR
  --weight_decay WEIGHT_DECAY
  --dropout DROPOUT
  --random_seed RANDOM_SEED
  --cpu
```
