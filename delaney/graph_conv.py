"""
Script that trains graph-conv models on Tox21 dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import numpy as np
import pandas as pd
np.random.seed(123)
import tensorflow as tf
tf.set_random_seed(123)
import deepchem as dc
from datasets import load_delaney
import timeit

# Load Delaney dataset
delaney_tasks, delaney_datasets, transformers = load_delaney(
    featurizer='GraphConv', split='random')
train_dataset, valid_dataset, test_dataset = delaney_datasets

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
# Dense post-processing layer

model = dc.models.MultitaskGraphRegressor(
    graph_model,
    len(delaney_tasks),
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

eval_time = timeit.default_timer() - start

print("Train scores")
print(train_score)

print("Validation scores")
print(valid_score)

if not os.path.exists('log/delaney'): os.makedirs('log/delaney')
out = open('log/delaney/graph_conv.log', 'w')
out.write('Train scores: %s\n' % train_score)
out.write('Validation scores: %s\n' % valid_score)
out.write('Train time: %.1fm\n' % (train_time/60.))
out.write('Eval time: %.1fm\n' % (eval_time/60.))
out.close()

scores = [
        train_scores['mean-pearson_r2_score'],
        valid_scores['mean-pearson_r2_score'],
        ]
scores = pd.DataFrame(scores).T
scores.columns = ['train','valid']
scores.to_pickle('log/delaney/graph_conv.pkl')
