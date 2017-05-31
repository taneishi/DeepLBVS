"""
Script that trains graph-conv models on ChEMBL dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import numpy as np
np.random.seed(123)
import pandas as pd
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc
from datasets import load_chembl
import timeit
import os

# Load ChEMBL dataset
chembl_tasks, datasets, transformers = load_chembl(
    shard_size=2000, featurizer="GraphConv", set="5thresh", split="random")
train_dataset, valid_dataset, test_dataset = datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# Do setup required for tf/keras models
# Number of features on conv-mols
n_feat = 75
# Batch size of models
batch_size = 128
graph_model = dc.nn.SequentialGraph(n_feat)
graph_model.add(dc.nn.GraphConv(128, n_feat, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphPool())
graph_model.add(dc.nn.GraphConv(128, 128, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphPool())
# Gather Projection
graph_model.add(dc.nn.Dense(256, 128, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))

model = dc.models.MultitaskGraphRegressor(
    graph_model,
    len(chembl_tasks),
    n_feat,
    batch_size=batch_size,
    learning_rate=0.0005,
    optimizer_type="adam",
    beta1=.9,
    beta2=.999)

start = timeit.default_timer()

# Fit trained model
model.fit(train_dataset, nb_epoch=20)

train_time = timeit.default_timer() - start

start = timeit.default_timer()

print("Evaluating model")
train_score, train_scores = model.evaluate(train_dataset, [metric], transformers, per_task_metrics=True)
valid_score, valid_scores = model.evaluate(valid_dataset, [metric], transformers, per_task_metrics=True)
test_score, test_scores = model.evaluate(test_dataset, [metric], transformers, per_task_metrics=True)

eval_time = timeit.default_timer() - start

print("Train scores")
print(train_score)

print("Validation scores")
print(valid_score)

print("Test scores")
print(test_score)


if not os.path.exists('log/chembl'): os.makedirs('log/chembl')
out = open('log/chembl/graph_conv.log', 'w')
out.write('Train scores: %s\n' % train_score)
out.write('Validation scores: %s\n' % valid_score)
out.write('Test scores: %s\n' % test_score)
out.write('Train time: %.1fm\n' % (train_time/60.))
out.write('Eval time: %.1fm\n' % (eval_time/60.))
out.close()

scores = [
        train_scores['mean-pearson_r2_score'],
        valid_scores['mean-pearson_r2_score'],
        test_scores['mean-pearson_r2_score'],
        ]
scores = pd.DataFrame(scores).T
scores.columns = ['train','valid','test']
scores.to_pickle('log/chembl/graph_conv.pkl')
