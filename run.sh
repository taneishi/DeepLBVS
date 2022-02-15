#!/bin/bash

if [ -d torch ]; then
    source torch/bin/activate
else
    python3 -m venv torch
    source torch/bin/activate
    pip install --upgrade pip
    pip install torch numpy scikit-learn xgboost pandas rdkit-pypi
fi

wget -c -P data https://github.com/deepchem/deepchem/raw/master/datasets/pcba.csv.gz

python pcba.py
python rf.py
python mlp.py
