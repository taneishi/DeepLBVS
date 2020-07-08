Deep Learning for Multi-modal Virtual Screening
===============================================

Dependency
----------

- PyTorch-1.5.0

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

Results
-------

```
epoch    0 batch    4/   4 train_loss  0.694 test_loss  0.693  1.47 sec
epoch    1 batch    4/   4 train_loss  0.693 test_loss  0.693  1.11 sec
epoch    2 batch    4/   4 train_loss  0.693 test_loss  0.693  1.26 sec
epoch    3 batch    4/   4 train_loss  0.692 test_loss  0.692  1.32 sec
epoch    4 batch    4/   4 train_loss  0.691 test_loss  0.692  1.27 sec
epoch    5 batch    4/   4 train_loss  0.691 test_loss  0.691  1.31 sec
epoch    6 batch    4/   4 train_loss  0.690 test_loss  0.691  1.27 sec
epoch    7 batch    4/   4 train_loss  0.689 test_loss  0.690  1.09 sec
epoch    8 batch    4/   4 train_loss  0.688 test_loss  0.689  1.34 sec
epoch    9 batch    4/   4 train_loss  0.686 test_loss  0.688  1.31 sec
epoch   10 batch    4/   4 train_loss  0.685 test_loss  0.689  1.31 sec
...

```
