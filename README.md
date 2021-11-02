Prediction of Compound-Protein Interactions
===========================================

Environment
-----------

```
python3 -m venv rdkit
source rdkit/bin/activate
pip install rdkit torch scikit-learn
```

- Python-3.8
- rdkit-2021.9.2
- scikit-learn-1.0.1
- torch-1.10.0

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
