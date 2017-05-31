"""
Script that trains graph-conv models on PCBA dataset.
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
from datasets import load_pcba
import timeit
import os

# Load PCBA dataset
pcba_tasks, datasets, transformers = load_pcba(
    featurizer="GraphConv", split="index")
train_dataset, valid_dataset, test_dataset = datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, np.mean)

# Do setup required for tf/keras models
# Number of features on conv-mols
n_feat = 75
# Batch size of models
batch_size = 50
graph_model = dc.nn.SequentialGraph(n_feat)
graph_model.add(dc.nn.GraphConv(64, n_feat, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphPool())
graph_model.add(dc.nn.GraphConv(64, 64, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphPool())
# Gather Projection
graph_model.add(dc.nn.Dense(128, 64, activation='relu'))
graph_model.add(dc.nn.BatchNormalization(epsilon=1e-5, mode=1))
graph_model.add(dc.nn.GraphGather(batch_size, activation="tanh"))

model = dc.models.MultitaskGraphClassifier(
    graph_model,
    len(pcba_tasks),
    n_feat,
    batch_size=batch_size,
    learning_rate=0.0005,
    optimizer_type="adam",
    beta1=.9,
    beta2=.999)

start = timeit.default_timer()

# Fit trained model
model.fit(train_dataset, nb_epoch=15)

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

if not os.path.exists('log/pcba'): os.makedirs('log/pcba')
out = open('log/pcba/graph_conv.log', 'w')
out.write('Train scores: %s\n' % train_score)
out.write('Validation scores: %s\n' % valid_score)
out.write('Test scores: %s\n' % test_score)
out.write('Train time: %.1fm\n' % (train_time/60.))
out.write('Eval time: %.1fm\n' % (eval_time/60.))
out.close()

scores = [
        train_scores['mean-roc_auc_score'],
        valid_scores['mean-roc_auc_score'],
        test_scores['mean-roc_auc_score'],
        ]
scores = pd.DataFrame(scores).T
scores.columns = ['train','valid','test']
scores.to_pickle('log/pcba/graph_conv.pkl')
