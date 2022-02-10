#!/bin/bash

if [ -d torch ]; then
    source torch/bin/activate
else
    python3 -m venv torch
    source torch/bin/activate
    pip install --upgrade pip
    pip install torch numpy scikit-learn pandas rdkit-pypi
fi

python pcba.py
