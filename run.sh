#!/bin/bash

pip install -r requirements.txt

mkdir -p data
wget -c -P data https://github.com/deepchem/deepchem/raw/master/datasets/pcba.csv.gz

python pcba.py
python rf.py
python mlp.py
