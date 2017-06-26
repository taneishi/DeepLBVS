"""
Script that trains multitask models on Delaney dataset.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import shutil
import numpy as np
import pandas as pd
import deepchem as dc
from datasets import load_delaney
import timeit

# Only for debug!
np.random.seed(123)

# Load Delaney dataset
delaney_tasks, delaney_datasets, transformers = load_delaney(
    featurizer='ECFP', split='random')
train_dataset, valid_dataset, test_dataset = delaney_datasets

# Fit models
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

n_layers = 2
nb_epoch = 50
model = dc.models.TensorflowMultiTaskRegressor(
    len(delaney_tasks), train_dataset.get_data_shape()[0],
    layer_sizes=[1000]*n_layers, dropouts=[0.25]*n_layers,
    weight_init_stddevs=[0.02]*n_layers,
    bias_init_consts=[1.]*n_layers, learning_rate=0.0008,
    penalty=0.0005, penalty_type="l2", optimizer="adam", batch_size=128,
    seed=123, verbosity="high")

start = timeit.default_timer()

# Fit trained model
model.fit(train_dataset, nb_epoch=nb_epoch)

train_time = timeit.default_timer() - start

model.save()

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
out = open('log/delaney/tf_models.log', 'w')
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
scores.to_pickle('log/delaney/tf_models.pkl')
